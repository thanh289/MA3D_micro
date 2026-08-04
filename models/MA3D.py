import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ThreeDMM_Adaptive import LandmarkModulationFusion, SpatialLandmarkModulationFusion
from .motion_encoder import MotionEncoderCNN
from .motion_encoder_rmt import RMTMotionEncoder
from .appearance_encoder import AppearanceEncoderViT
from .rise_fall_fusion import RiseFallAgreementFusion


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    return model


class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, target_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y_hat = self.linear(x)
        return y_hat


class MA3D(nn.Module):
    """
    New architecture direction (ME). The three branches are re-purposed
    instead of removed:

      - Motion branch (flow, MotionEncoderCNN, trained from scratch):
        the MAIN CONTENT branch (used to be the IR50-on-apex role in the
        original MaE architecture). Fed with the 3-ROI optical-flow map
        (eyebrow, eye, mouth).

      - Landmark branch (MobileFaceNet, frozen): the FiLM MODULATOR (used
        to be the 3DMM-parameter role). Fed with APEX -- apex is when the
        AU activation is most expressed, giving the most informative
        "where on the face is this happening" signal. Does not
        participate in cross-attention directly.

        Route B (decided): modulation is now SPATIAL, not
        channel-only. The original FiLM (kept in ThreeDMM_Adaptive.py as
        LandmarkModulationFusion, for comparison) pools the landmark map
        down to a single vector and broadcasts (gamma, beta) uniformly
        over the whole 7x7 motion map -- which cannot express "amplify
        THIS region, not that one", even though that's exactly what this
        branch is supposed to be doing. SpatialLandmarkModulationFusion
        instead keeps the landmark map's 7x7 spatial grid (it's already
        aligned 1:1 with the motion map's grid, no resize needed) and
        produces a PER-POSITION (gamma, beta).

      - Appearance-context branch: the CROSS-ATTENTION partner (used to be
        the landmark's role in the original MaE architecture). Fed with
        ONSET -- a neutral reference frame, avoiding redundancy with the
        AU-expressed pose already carried by the motion+landmark branches.

        Route C1 (decided): no longer a frozen, full-depth IR50.
        IR50 here was trained under an ArcFace objective, whose explicit
        goal is to become INVARIANT to expression (keep identity, discard
        everything else) -- using it, frozen, to supply "where is this
        expression happening" context works against what it was trained
        to do, and that risk is larger for ME (an already-weak signal)
        than it was in the MaE-domain original. AppearanceEncoderViT is a
        shallow (depth=2), TRAINED-FROM-SCRATCH transformer instead,
        borrowing GAMDSS's "vit_pos" design: the raw onset frame is
        bilinearly downsampled directly to a 7x7 grid (no conv stem),
        patchified with patch_size=1, then run through 2 Transformer
        blocks -- deliberately shallow so it can't re-learn identity, only
        coarse positional structure (same rationale TSFmicro and GAMDSS
        both use for their equivalent "static/context" branch).

    pyramid_fuse (bidirectional cross-attention), SE_block and
    ClassificationHead are UNCHANGED -- Route B/C1 only swap what feeds
    into them, not the fusion mechanism itself.

    Rise/fall dual-branch motion (use_rise_fall, decided in chat). The
    motion branch above can optionally be fed TWO flow maps -- rise-phase
    (onset->apex) AND fall-phase (apex->offset) -- through the SAME
    MotionEncoderCNN instance (shared weights, called twice). How the two
    phases get combined into a final prediction is controlled by
    rise_fall_mode, since neither reference paper's ACTUAL code (verified
    against CausalNet's and GAMDSS's real source, not just their paper
    text) agrees on a single "correct" mechanism -- see chat notes:

      - rise_fall_mode="feature_gate" (this codebase's own design, NOT a
        faithful port of either paper): X_rise, X_fall combined via
        RiseFallAgreementFusion (rise_fall_fusion.py) into ONE fused
        motion representation BEFORE landmark FiLM + pyramid_fuse -- i.e.
        rise/fall fusion and landmark FiLM are STACKED, landmark FiLM
        itself unchanged. Two small auxiliary classification heads
        (aux_head_rise, aux_head_fall) are applied to each phase's pooled
        feature BEFORE fusion, for extra gradient signal. pyramid_fuse
        runs ONCE (cheaper).

      - rise_fall_mode="decision_level" (matches GAMDSS's ACTUAL BDRT class
        AND its real training script, verified against real source): NO
        feature-level fusion at all. landmark FiLM + pyramid_fuse +
        se_block + head (all shared-weight instances) run TWICE, once per
        phase, producing out_rise and out_fall as two INDEPENDENT full
        logit vectors -- exactly how GAMDSS's BDRT.forward() returns (out,
        out_b). The final prediction `out` is out_rise ONLY -- the
        fall-phase output out_fall is used SOLELY as an auxiliary loss
        term (aux["fall"]), never averaged into the prediction. This
        matches the actual GAMDSS training script exactly: `ALL, s =
        net(...)`; `loss = CE(ALL,y) + CE(s,y)`; `predicts =
        torch.max(ALL, 1)` -- s/out_fall never touches the prediction.
        pyramid_fuse runs TWICE (roughly 2x the compute of feature_gate
        mode) even though only one of the two outputs is actually used as
        the prediction -- the second pass exists purely to generate a
        useful gradient signal for the shared weights via the auxiliary
        loss.

    AU-guidance branch (use_au, added inspired by STAG's "AU Guidance"
    step -- STAG itself has no public code, this is our own design, not a
    port): OPTIONAL side branch, orthogonal to everything above (works
    with any use_rise_fall/rise_fall_mode/motion_backbone combination).

      x_au [B, au_dim] (au_dim=36 by default: 18 dynamic-AU one-hot +
      18 static-AU "(k)" one-hot slots, see au_utils.py for the parsing
      convention and the exact vocab) -> au_encoder (2-layer MLP, matches
      STAG's own "binary AU vector -> MLP -> f_au" design) -> au_embed
      [B, au_embed_dim] -> CONCATENATED onto the pooled fusion feature
      (post se_block, pre head) -> head's input_dim becomes
      512 + au_embed_dim instead of 512.

      Applied inside _film_and_classify (the shared tail), so in
      rise_fall_mode="decision_level" the SAME au_embed (computed once in
      forward(), not per-phase -- AU annotation is a property of the
      whole clip, not of rise vs. fall separately) is concatenated on
      BOTH the rise-phase and fall-phase passes.

      When use_au=False (default), behaves exactly as before -- x_au is
      simply ignored, head's input_dim stays 512.

    NOTE for rise_fall_mode="decision_level": GAMDSS's real BDRT also
      sources the position-calibration branch from a DIFFERENT frame per
      phase (onset for rise, apex for fall). This codebase does NOT
      replicate that -- the landmark branch stays fixed on APEX and the
      appearance branch stays fixed on ONSET for BOTH phases, only the
      MOTION input changes per pass. This keeps rise_fall_mode a
      cleanly-scoped ablation of "how to combine rise+fall" without also
      conflating a second, separate change (which frame feeds the
      landmark/appearance branches per phase).

    When use_rise_fall=False, forward() behaves exactly like the original
    single-phase (rise-only) version regardless of rise_fall_mode.

    motion_backbone (decided in chat, GAMDSS-inspired): swaps WHAT encodes
    motion, orthogonal to use_rise_fall/rise_fall_mode above (works with
    either):

      - motion_backbone="cnn" (default, unchanged): MotionEncoderCNN,
        trained from scratch, operating on the 3-ROI-cropped flow map
        (x_flow_rise / x_flow_fall, [B, n_roi, 3, H, W] -- eyebrow, eye,
        mouth crops from run_inference_flow.py).

      - motion_backbone="rmt": RMTMotionEncoder (motion_encoder_rmt.py),
        wrapping GAMDSS's real RMT_T3/VisRetNet retention-based backbone
        (RMT.py, copied verbatim from GAMDSS's source). Operates on a
        WHOLE-FACE pixel-diff instead of ROI crops -- GAMDSS's real
        architecture has no ROI-cropping concept at all -- computed ON THE
        FLY inside forward() from the already-available RGB frames
        (rise: x_apex - x_onset; fall: x_offset - x_apex, x_offset being a
        NEW optional forward() argument), NOT from flow_map.npy/
        flow_map_fall.npy at all (those are ignored when motion_backbone=
        "rmt", though still required as positional args for interface
        compatibility -- pass anything of the right shape, e.g. zeros).
    """

    def __init__(self, img_size=224, num_classes=7, type="large",
                 n_roi=3, landmark_embed_dim=256, num_film_blocks=5,
                 use_spatial_film=True, use_rise_fall=True,
                 rise_fall_mode="feature_gate", motion_backbone="cnn",
                 use_au=False, au_dim=36, au_embed_dim=128):
        super().__init__()
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.img_size = img_size
        self.num_classes = num_classes
        self.use_rise_fall = use_rise_fall

        assert rise_fall_mode in ("feature_gate", "decision_level"), \
            f"rise_fall_mode must be 'feature_gate' or 'decision_level', got {rise_fall_mode!r}"
        self.rise_fall_mode = rise_fall_mode

        assert motion_backbone in ("cnn", "rmt"), \
            f"motion_backbone must be 'cnn' or 'rmt', got {motion_backbone!r}"
        self.motion_backbone = motion_backbone

        self.use_au = use_au

        # --- Landmark branch (frozen) -- FiLM modulator source ---
        self.face_landback = MobileFaceNet([112, 112], 136)
        face_landback_checkpoint = torch.load(
            "checkpoints/mobilefacenet_model_best.pth_o1.tar",
            map_location=lambda storage, loc: storage)
        self.face_landback.load_state_dict(face_landback_checkpoint['state_dict'])

        for param in self.face_landback.parameters():
            param.requires_grad = False

        # --- Appearance-context branch (Route C1: shallow ViT, trained from
        # scratch -- no checkpoint, no freezing; see class docstring) ---
        self.appearance_encoder = AppearanceEncoderViT(
            grid_size=7, embed_dim=512, depth=2, num_heads=4,
        )

        # --- Motion branch (trained from scratch) -- main content ---
        # SHARED weights: when use_rise_fall=True, this SAME instance is
        # called TWICE in forward() (once on flow_rise, once on flow_fall)
        # -- this IS the "share-weight" design agreed in chat, not two
        # separate encoders.
        if motion_backbone == "cnn":
            self.motion_encoder = MotionEncoderCNN(n_roi=n_roi, out_channels=512,
                                        deterministic_pool=True)
        else:  # "rmt"
            self.motion_encoder = RMTMotionEncoder()

        if use_rise_fall and rise_fall_mode == "feature_gate":
            self.rise_fall_fusion = RiseFallAgreementFusion()
            # Auxiliary per-phase classification heads (extra gradient
            # signal -- see class docstring). GAP over the 7x7 grid, then a
            # plain linear head; deliberately as cheap as possible given
            # how little data there is. Only needed in feature_gate mode --
            # decision_level mode gets its "per-phase logits" for free from
            # running the full head twice, no separate small heads needed.
            self.aux_head_rise = nn.Linear(512, num_classes)
            self.aux_head_fall = nn.Linear(512, num_classes)

        # --- Landmark-driven FiLM modulation of the motion feature map ---
        self.use_spatial_film = use_spatial_film
        if use_spatial_film:
            # Route B: spatial (per-position) gamma/beta, see class docstring.
            self.landmark_fusion = SpatialLandmarkModulationFusion(
                feat_nc=512,
                landmark_dim=512,
                num_blocks=num_film_blocks,
                hidden_dim=landmark_embed_dim // 2,
            )
        else:
            # Original channel-only FiLM, kept available for comparison.
            self.landmark_fusion = LandmarkModulationFusion(
                feat_nc=512,
                embed_dim=landmark_embed_dim,
                num_blocks=num_film_blocks,
                landmark_dim=512,
            )

        # --- Cross-attention + classification head (unchanged) ---
        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel=49, embed_dim=512,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        self.se_block = SE_block(input_dim=512)

        # --- Optional AU-guidance branch (STAG-inspired, our own design) ---
        # 2-layer MLP over the multi-hot AU vector -> au_embed, concatenated
        # onto the pooled fusion feature right before the head. See class
        # docstring and au_utils.py for the vector convention.
        head_input_dim = 512
        if use_au:
            self.au_encoder = nn.Sequential(
                nn.Linear(au_dim, au_embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(au_embed_dim, au_embed_dim),
            )
            head_input_dim = 512 + au_embed_dim

        self.head = ClassificationHead(input_dim=head_input_dim, target_dim=self.num_classes)

    def _film_and_classify(self, x_motion, x_lmk_map, x_appear, B, au_embed=None):
        """
        Shared tail of the pipeline: landmark FiLM modulation ->
        pyramid_fuse (bidirectional cross-attention) -> se_block ->
        [optional AU concat] -> head.
        Factored out so rise_fall_mode="decision_level" can call this
        TWICE (once per phase, same weights both times) without duplicating
        the logic -- this IS what makes decision_level mode a faithful
        match to GAMDSS's real BDRT (one shared backbone/head, called
        independently per phase, see class docstring).

        au_embed: [B, au_embed_dim] or None. When not None, the SAME
        au_embed is concatenated on every call -- AU annotation is a
        whole-clip property, so decision_level mode's two calls (rise
        pass, fall pass) both get the identical au_embed, not a
        phase-specific one.

        Returns: out [B, num_classes], y_feat [B, 512 (+au_embed_dim)], attn_all
        """
        if self.use_spatial_film:
            x_motion = self.landmark_fusion(x_motion, x_lmk_map)    # [B, 512, 7, 7], spatial gamma/beta
        else:
            x_lmk_tokens = x_lmk_map.view(B, -1, 49).transpose(1, 2)  # [B, 49, 512]
            x_motion = self.landmark_fusion(x_motion, x_lmk_tokens)   # [B, 512, 7, 7], channel-only gamma/beta
        x_motion = x_motion.view(B, 512, -1).transpose(1, 2)        # [B, 49, 512]

        # NOTE: pyramid_fuse(a, b) only returns the CLS token of the FIRST
        # argument (see hyp_crossvit.py -- x_class2 is computed but never
        # returned), so argument order matters: motion (the branch we want
        # driving the final decision) goes first, appearance-context second.
        y_hat, attn_all = self.pyramid_fuse(x_motion, x_appear)
        y_hat = self.se_block(y_hat)

        if au_embed is not None:
            y_hat = torch.cat([y_hat, au_embed], dim=-1)  # [B, 512+au_embed_dim]

        y_feat = y_hat
        out = self.head(y_hat)
        return out, y_feat, attn_all

    def forward(self, x_apex, x_onset, x_flow_rise, x_flow_fall=None, x_offset=None,
                x_au=None):
        """
        x_apex      : [B, 3, 224, 224] RGB apex frame  -> landmark/modulator branch,
                      also used as rise-phase diff source when motion_backbone="rmt"
        x_onset     : [B, 3, 224, 224] RGB onset frame -> appearance-context branch,
                      also used as rise-phase diff source when motion_backbone="rmt"
        x_flow_rise : [B, n_roi, 3, H, W] onset->apex flow map (u, v, strain
                      per ROI) -> motion branch. IGNORED when
                      motion_backbone="rmt" (still required positionally,
                      any correctly-shaped tensor works, e.g. zeros).
        x_flow_fall : [B, n_roi, 3, H, W] apex->offset flow map, SAME shape
                      as x_flow_rise -> motion branch, SHARED weights.
                      REQUIRED when self.use_rise_fall=True AND
                      motion_backbone="cnn"; IGNORED when motion_backbone=
                      "rmt" (x_offset is used instead, see below).
        x_offset    : [B, 3, 224, 224] RGB offset frame. ONLY used when
                      motion_backbone="rmt" AND self.use_rise_fall=True
                      (to compute the fall-phase whole-face diff,
                      x_offset - x_apex) -- unused/ignored otherwise.
        x_au        : [B, au_dim] multi-hot AU vector (see au_utils.py).
                      REQUIRED when self.use_au=True; ignored/unused when
                      self.use_au=False (still fine to pass None then).

        Returns: out, y_feat, attn_all, aux
            aux = {
                "rise": [B, num_classes] logits from the rise phase alone
                         (feature_gate mode's small aux head output; ALWAYS
                         None in decision_level mode -- see class docstring
                         on why, to avoid double-counting `out` itself),
                "fall": [B, num_classes] logits from the fall phase alone
                         (feature_gate mode's small aux head output, OR
                         decision_level mode's full out_fall -- the ONLY
                         auxiliary term in that mode), or None if
                         use_rise_fall=False,
                "agreement": [B, 1, 7, 7] rise/fall agreement map
                         (feature_gate mode only), or None otherwise,
            }
        """
        B = x_apex.shape[0]

        # --- Landmark branch (frozen), fed with APEX ---
        x_apex_112 = F.interpolate(x_apex, size=112)
        _, x_lmk_map = self.face_landback(x_apex_112)               # [B, 512, 7, 7]

        # --- Appearance-context branch (Route C1: shallow ViT, trained from
        # scratch), fed with ONSET ---
        x_appear = self.appearance_encoder(x_onset)                 # [B, 49, 512]

        aux = {"rise": None, "fall": None, "agreement": None}

        # --- Optional AU-guidance branch: computed ONCE (AU annotation is
        # a whole-clip property, not rise- or fall-specific), reused by
        # every _film_and_classify() call below regardless of
        # use_rise_fall/rise_fall_mode. ---
        au_embed = None
        if self.use_au:
            assert x_au is not None, (
                "self.use_au=True requires x_au [B, au_dim] to be passed to forward()"
            )
            au_embed = self.au_encoder(x_au)                        # [B, au_embed_dim]

        # --- Motion branch input: either the ROI-cropped flow map (cnn) or
        # a whole-face RGB diff computed on the fly (rmt) -- see class
        # docstring on motion_backbone.
        if self.motion_backbone == "cnn":
            motion_input_rise = x_flow_rise
            motion_input_fall = x_flow_fall
        else:  # "rmt"
            motion_input_rise = x_apex - x_onset
            motion_input_fall = (x_offset - x_apex) if x_offset is not None else None

        if not self.use_rise_fall:
            # Original single-phase (rise-only) behaviour, unchanged.
            x_motion = self.motion_encoder(motion_input_rise)        # [B, 512, 7, 7]
            out, y_feat, attn_all = self._film_and_classify(
                x_motion, x_lmk_map, x_appear, B, au_embed=au_embed)
            return out, y_feat, attn_all, aux

        assert motion_input_fall is not None, (
            "use_rise_fall=True requires x_flow_fall (motion_backbone='cnn') "
            "or x_offset (motion_backbone='rmt')"
        )

        # SAME motion_encoder instance called twice -- shared weights,
        # regardless of rise_fall_mode or motion_backbone.
        x_motion_rise = self.motion_encoder(motion_input_rise)        # [B, 512, 7, 7]
        x_motion_fall = self.motion_encoder(motion_input_fall)        # [B, 512, 7, 7]

        if self.rise_fall_mode == "feature_gate":
            # Auxiliary per-phase classification (small heads, on pooled
            # features, BEFORE fusion) -- see class docstring.
            aux["rise"] = self.aux_head_rise(x_motion_rise.mean(dim=[2, 3]))
            aux["fall"] = self.aux_head_fall(x_motion_fall.mean(dim=[2, 3]))

            x_motion, aux["agreement"] = self.rise_fall_fusion(x_motion_rise, x_motion_fall)
            out, y_feat, attn_all = self._film_and_classify(
                x_motion, x_lmk_map, x_appear, B, au_embed=au_embed)
            return out, y_feat, attn_all, aux

        else:  # rise_fall_mode == "decision_level"
            # NO feature-level fusion -- the shared FiLM+pyramid_fuse+head
            # tail runs TWICE, once per phase (matches GAMDSS's real BDRT:
            # main_branch(act, POS) called independently for rise and
            # fall -- verified against the actual GAMDSS training script,
            # not just model.py/RMT.py: `ALL, s = net(...)`.
            out_rise, feat_rise, attn_rise = self._film_and_classify(
                x_motion_rise, x_lmk_map, x_appear, B, au_embed=au_embed)
            out_fall, feat_fall, attn_fall = self._film_and_classify(
                x_motion_fall, x_lmk_map, x_appear, B, au_embed=au_embed)

            # IMPORTANT (corrected after seeing GAMDSS's real training
            # loop): the two outputs are NOT averaged. GAMDSS's script
            # does `ALL, s = net(...)`, `loss = CE(ALL,y) + CE(s,y)`, but
            # `predicts = torch.max(ALL, 1)` -- ONLY the rise-phase output
            # (ALL / out_rise) is ever used as the actual prediction. The
            # fall-phase output (s / out_fall) contributes PURELY as an
            # auxiliary loss term that shapes the shared backbone's
            # weights, never as part of the prediction itself.
            out = out_rise
            y_feat = feat_rise

            # aux["rise"] intentionally left None here (not aux["fall"]-
            # symmetric like feature_gate mode): out_rise IS `out` above,
            # already covered by the main loss (MACE/warmup on `logits` in
            # engine.py) -- setting aux["rise"] = out_rise too would
            # double-count the SAME tensor's CE loss twice with no
            # corresponding behaviour in GAMDSS's real script (which only
            # computes CE(ALL) once). aux["fall"] is the one and only
            # extra term, matching GAMDSS's `loss_s = CE(s, y)`.
            aux["rise"] = None
            aux["fall"] = out_fall

            return out, y_feat, attn_rise, aux