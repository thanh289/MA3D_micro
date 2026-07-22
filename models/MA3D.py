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
from .appearance_encoder import AppearanceEncoderViT


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
    """

    def __init__(self, img_size=224, num_classes=7, type="large",
                 n_roi=3, landmark_embed_dim=256, num_film_blocks=5,
                 use_spatial_film=True):
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
        self.motion_encoder = MotionEncoderCNN(n_roi=n_roi, out_channels=512)

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
        self.head = ClassificationHead(input_dim=512, target_dim=self.num_classes)

    def forward(self, x_apex, x_onset, x_flow):
        """
        x_apex  : [B, 3, 224, 224] RGB apex frame  -> landmark/modulator branch
        x_onset : [B, 3, 224, 224] RGB onset frame -> appearance-context branch
        x_flow  : [B, n_roi, 3, H, W] flow map (u, v, strain per ROI) -> motion branch
        """
        B = x_apex.shape[0]

        # --- Landmark branch (frozen), fed with APEX ---
        x_apex_112 = F.interpolate(x_apex, size=112)
        _, x_lmk_map = self.face_landback(x_apex_112)               # [B, 512, 7, 7]

        # --- Appearance-context branch (Route C1: shallow ViT, trained from
        # scratch), fed with ONSET ---
        x_appear = self.appearance_encoder(x_onset)                 # [B, 49, 512]

        # --- Motion branch (trained from scratch), fed with the flow map ---
        x_motion = self.motion_encoder(x_flow)                      # [B, 512, 7, 7]

        # --- Landmark-driven FiLM modulation of the motion feature map ---
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

        y_feat = y_hat
        out = self.head(y_hat)

        return out, y_feat, attn_all