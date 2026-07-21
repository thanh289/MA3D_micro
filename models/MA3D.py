import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ir50 import Backbone
from .ThreeDMM_Adaptive import LandmarkModulationFusion
from .motion_encoder import MotionEncoderCNN


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
        now the MAIN CONTENT branch (used to be the IR50-on-apex role).
        Fed with the 3-ROI optical-flow map (eyebrow, eye, mouth).

      - Landmark branch (MobileFaceNet, frozen): now the FiLM MODULATOR
        (used to be the 3DMM-parameter role). Fed with APEX -- apex is
        when the AU activation is most expressed, giving the most
        informative "where on the face is this happening" signal.
        No longer participates in cross-attention directly.

      - Appearance-context branch (IR50, frozen): now the CROSS-ATTENTION
        partner (used to be the landmark's role). Fed with ONSET -- a
        neutral reference frame, avoiding redundancy with the AU-expressed
        pose already carried by the motion+landmark branches.

    pyramid_fuse, SE_block and ClassificationHead are unchanged from the
    original MaE architecture.
    """

    def __init__(self, img_size=224, num_classes=7, type="large",
                 n_roi=3, landmark_embed_dim=256, num_film_blocks=5):
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

        # --- Appearance-context branch (frozen) -- cross-attention partner ---
        self.ir_back = Backbone(50, 0.0, 'ir')
        ir_checkpoint = torch.load("checkpoints/ir50_o1.pth",
                                   map_location=lambda storage, loc: storage)
        self.ir_back = load_pretrained_weights(self.ir_back, ir_checkpoint)

        for param in self.ir_back.parameters():
            param.requires_grad = False

        self.ir_layer = nn.Linear(1024, 512)

        # --- Motion branch (trained from scratch) -- main content ---
        self.motion_encoder = MotionEncoderCNN(n_roi=n_roi, out_channels=512)

        # --- Landmark-driven FiLM modulation of the motion feature map ---
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
        x_lmk_tokens = x_lmk_map.view(B, -1, 49).transpose(1, 2)    # [B, 49, 512]

        # --- Appearance-context branch (frozen), fed with ONSET ---
        x_appear = self.ir_back(x_onset).view(B, 49, 1024)
        x_appear = self.ir_layer(x_appear)                          # [B, 49, 512]

        # --- Motion branch (trained from scratch), fed with the flow map ---
        x_motion = self.motion_encoder(x_flow)                      # [B, 512, 7, 7]

        # --- Landmark-driven FiLM modulation of the motion feature map ---
        x_motion = self.landmark_fusion(x_motion, x_lmk_tokens)     # [B, 512, 7, 7]
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