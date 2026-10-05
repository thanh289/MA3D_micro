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
from .roi_alignment import sample_rois, ROIContextAggregator


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
    """Three-stream ROI model for CNN motion; legacy full-face RMT ablation.

    CNN: independent motion ROIs, apex geometry sampled using the SAME
    normalized boxes, and onset crops encoded independently. Local FiLM
    and cross-interaction precede attention across ROI summaries.
    decision_level uses fall only for the training objective.
    Existing optimization and constructor parameters are retained.
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

        self.face_landback.eval()

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
        self.roi_aggregator = ROIContextAggregator(n_roi, 512) if motion_backbone == "cnn" else None

    def _legacy_film_and_classify(self, x_motion, x_lmk_map, x_appear, B, au_embed=None):
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

    def _legacy_forward(self, x_apex, x_onset, x_flow_rise, x_flow_fall=None, x_offset=None,
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
        # every _legacy_film_and_classify() call below regardless of
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

        if not self.use_rise_fall or (not self.training and self.rise_fall_mode == "decision_level"):
            # Original single-phase (rise-only) behaviour, unchanged.
            x_motion = self.motion_encoder(motion_input_rise)        # [B, 512, 7, 7]
            out, y_feat, attn_all = self._legacy_film_and_classify(
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
            out, y_feat, attn_all = self._legacy_film_and_classify(
                x_motion, x_lmk_map, x_appear, B, au_embed=au_embed)
            return out, y_feat, attn_all, aux

        else:  # rise_fall_mode == "decision_level"
            # NO feature-level fusion -- the shared FiLM+pyramid_fuse+head
            # tail runs TWICE, once per phase (matches GAMDSS's real BDRT:
            # main_branch(act, POS) called independently for rise and
            # fall -- verified against the actual GAMDSS training script,
            # not just model.py/RMT.py: `ALL, s = net(...)`.
            out_rise, feat_rise, attn_rise = self._legacy_film_and_classify(
                x_motion_rise, x_lmk_map, x_appear, B, au_embed=au_embed)
            out_fall, feat_fall, attn_fall = self._legacy_film_and_classify(
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

    def train(self, mode=True):
        """Freeze parameters AND BatchNorm buffers on the landmark backbone."""
        super().train(mode)
        self.face_landback.eval()
        return self

    def _classify_rois(self, motion, geometry, appearance, boxes, au_embed):
        B, R, C, H, W = motion.shape
        motion = motion.reshape(B * R, C, H, W)
        geometry = geometry.reshape(B * R, C, H, W)
        if self.use_spatial_film:
            motion = self.landmark_fusion(motion, geometry)
        else:
            motion = self.landmark_fusion(motion, geometry.flatten(2).transpose(1, 2))
        local, attn = self.pyramid_fuse(motion.flatten(2).transpose(1, 2), appearance)
        local = local.reshape(B, R, C)
        fused = self.roi_aggregator(local, boxes)
        fused = self.se_block(fused)
        if au_embed is not None:
            fused = torch.cat([fused, au_embed], dim=-1)
        return self.head(fused), fused, attn

    def forward(self, x_apex, x_onset, x_flow_rise, x_flow_fall=None,
                x_offset=None, x_au=None, roi_boxes=None):
        if self.motion_backbone != "cnn":
            return self._legacy_forward(x_apex, x_onset, x_flow_rise,
                                        x_flow_fall, x_offset, x_au)
        if roi_boxes is None:
            raise ValueError("ROI CNN requires roi_boxes [B,R,4] from preprocessing; "
                             "regenerate metadata instead of assuming full-face alignment")
        B, R = x_flow_rise.shape[:2]
        if roi_boxes.shape != (B, R, 4):
            raise ValueError(f"roi_boxes must have shape {(B, R, 4)}")
        with torch.no_grad():
            _, full_geometry = self.face_landback(F.interpolate(x_apex, size=112))
        geometry = sample_rois(full_geometry, roi_boxes, (7, 7)).reshape(B, R, 512, 7, 7)
        # Crop BEFORE the existing shallow appearance encoder downsamples to 7x7.
        onset_rois = sample_rois(x_onset, roi_boxes, x_onset.shape[-2:])
        appearance = self.appearance_encoder(onset_rois)
        au_embed = None
        if self.use_au:
            if x_au is None:
                raise ValueError("AU-enabled model requires x_au")
            au_embed = self.au_encoder(x_au)
        rise = self.motion_encoder.forward_rois(x_flow_rise)
        aux = {"rise": None, "fall": None, "agreement": None}
        # At evaluation, decision_level has no dependency on fall input.
        if not self.use_rise_fall or (not self.training and self.rise_fall_mode == "decision_level"):
            out, feat, attn = self._classify_rois(rise, geometry, appearance, roi_boxes, au_embed)
            return out, feat, attn, aux
        if x_flow_fall is None:
            raise ValueError("Fall motion required during dual-phase training / feature_gate eval")
        fall = self.motion_encoder.forward_rois(x_flow_fall)
        if self.rise_fall_mode == "feature_gate":
            aux["rise"] = self.aux_head_rise(rise.mean(dim=(1, 3, 4)))
            aux["fall"] = self.aux_head_fall(fall.mean(dim=(1, 3, 4)))
            fused, agreement = self.rise_fall_fusion(rise.flatten(0, 1), fall.flatten(0, 1))
            aux["agreement"] = agreement.reshape(B, R, 1, 7, 7)
            out, feat, attn = self._classify_rois(fused.reshape_as(rise), geometry, appearance, roi_boxes, au_embed)
        else:
            out, feat, attn = self._classify_rois(rise, geometry, appearance, roi_boxes, au_embed)
            aux["fall"], _, _ = self._classify_rois(fall, geometry, appearance, roi_boxes, au_embed)
        return out, feat, attn, aux
