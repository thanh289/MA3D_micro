import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from .hyp_crossvit import *
from .mobilefacenet import MobileFaceNet
from .ThreeDMM_Adaptive import LandmarkModulationFusion, SpatialLandmarkModulationFusion
from .motion_encoder import MotionEncoderCNN
from .appearance_encoder import AppearanceEncoderViT
from .rise_fall_fusion import RiseFallAgreementFusion
from .fullface_context import FullFaceGeometryContext, roi_token_coordinates


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
    """ROI motion tokens with full-face geometry and onset appearance.

    Geometry cross-attention yields a condition per motion token. FiLM keeps
    local ROI convolutions; one shared pyramid processes the concatenated
    motion sequence against full-face appearance. No final ROI aggregator.
    """
    def __init__(self, img_size=224, num_classes=7, type="large",
                 n_roi=3, landmark_embed_dim=256, num_film_blocks=5,
                 use_spatial_film=True, use_rise_fall=True,
                 rise_fall_mode="feature_gate", motion_backbone="cnn",
                 use_au=False, au_dim=36, au_embed_dim=128,
                 roi_pool_size=3, geometry_heads=4,
                 geometry_conditioning="cross_attention", use_roi_position=True,
                 appearance_fusion="cross_attention"):
        super().__init__()
        if motion_backbone != 'cnn':
            raise ValueError('This ROI-token architecture supports motion_backbone=cnn only')
        if roi_pool_size not in (2, 3, 4, 5, 7):
            raise ValueError('roi_pool_size must be 2, 3, 4, 5 or 7 (FiLM uses spatial InstanceNorm)')
        if n_roi < 1 or num_film_blocks < 1:
            raise ValueError('n_roi and num_film_blocks must be positive')
        if geometry_conditioning not in ('cross_attention', 'global_mean', 'none'):
            raise ValueError('Unknown geometry_conditioning')
        if not use_spatial_film and geometry_conditioning == 'cross_attention':
            raise ValueError('Use geometry_conditioning=global_mean for channel-only FiLM')
        if appearance_fusion not in ('cross_attention', 'none'):
            raise ValueError('appearance_fusion must be cross_attention or none')
        self.appearance_fusion = appearance_fusion
        self.n_roi = n_roi
        self.roi_pool_size = roi_pool_size
        self.geometry_conditioning = geometry_conditioning
        self.use_roi_position = use_roi_position
        depth = 8
        if type == "small":
            depth = 4
        if type == "base":
            depth = 6
        if type == "large":
            depth = 8

        self.model_config = dict(img_size=img_size, num_classes=num_classes, type=type,
            n_roi=n_roi, landmark_embed_dim=landmark_embed_dim, num_film_blocks=num_film_blocks,
            use_spatial_film=use_spatial_film, use_rise_fall=use_rise_fall,
            rise_fall_mode=rise_fall_mode, motion_backbone=motion_backbone,
            use_au=use_au, au_dim=au_dim, au_embed_dim=au_embed_dim,
            roi_pool_size=roi_pool_size, geometry_heads=geometry_heads,
            geometry_conditioning=geometry_conditioning, use_roi_position=use_roi_position,
            appearance_fusion=appearance_fusion)
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
        self.appearance_encoder = (AppearanceEncoderViT(
            grid_size=7, embed_dim=512, depth=2, num_heads=4,
        ) if appearance_fusion == "cross_attention" else None)

        # --- Motion branch (trained from scratch) -- main content ---
        # SHARED weights: when use_rise_fall=True, this SAME instance is
        # called TWICE in forward() (once on flow_rise, once on flow_fall)
        # -- this IS the "share-weight" design agreed in chat, not two
        # separate encoders.
        if motion_backbone == "cnn":
            self.motion_encoder = MotionEncoderCNN(n_roi=n_roi, out_channels=512,
                                        deterministic_pool=True, roi_pool_size=roi_pool_size)

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
        if geometry_conditioning == 'none':
            self.landmark_fusion = None
        elif geometry_conditioning == 'cross_attention':
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

        # --- Query-aligned pyramid: motion R*g*g tokens, full-face appearance 49 ---
        self.pyramid_fuse = (HyVisionTransformer(in_chans=n_roi * roi_pool_size ** 2, q_chanel=49, embed_dim=512,
                                             query_aligned=True,
                                             depth=depth, num_heads=8, mlp_ratio=2.,
                                             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)
                             if appearance_fusion == "cross_attention" else None)

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
        self.geometry_context = (FullFaceGeometryContext(512, geometry_heads, use_roi_position)
                                 if geometry_conditioning == 'cross_attention' else None)
        self.motion_position = (nn.Linear(2, 512) if use_roi_position and
                                self.geometry_context is None else None)

    def train(self, mode=True):
        """Freeze parameters AND BatchNorm buffers on the landmark backbone."""
        super().train(mode)
        self.face_landback.eval()
        return self

    def _classify_rois(self, motion, geometry, appearance, boxes, au_embed):
        B, R, C, G, _ = motion.shape
        tokens = motion.flatten(3).permute(0, 1, 3, 2).reshape(B, R * G * G, C)
        xy = roi_token_coordinates(boxes.to(device=tokens.device, dtype=tokens.dtype), G)
        local = motion.reshape(B * R, C, G, G)
        if self.geometry_conditioning == 'cross_attention':
            context = self.geometry_context(tokens, geometry, xy)
            context = context.reshape(B, R, G, G, C).permute(0, 1, 4, 2, 3)
            local = self.landmark_fusion(local, context.reshape(B * R, C, G, G))
        elif self.geometry_conditioning == 'global_mean':
            face = geometry.flatten(2).transpose(1, 2)
            face = face[:, None].expand(B, R, -1, C).reshape(B * R, -1, C)
            local = self.landmark_fusion(local, face)
        tokens = local.reshape(B, R, C, G * G).permute(0, 1, 3, 2).reshape(B, -1, C)
        if self.use_roi_position:
            position = (self.geometry_context.position if self.geometry_context is not None
                        else self.motion_position)
            tokens = tokens + position(xy)
        if self.appearance_fusion == 'none':
            # Pool motion tokens after FiLM and the existing coordinate encoding.
            # No appearance encoder or pyramid parameters exist in this ablation.
            fused, attn = tokens.mean(dim=1), []
        else:
            fused, attn = self.pyramid_fuse(tokens, appearance)
        fused = self.se_block(fused)
        if au_embed is not None:
            fused = torch.cat([fused, au_embed], dim=-1)
        return self.head(fused), fused, attn

    def forward(self, x_apex, x_onset, x_flow_rise, x_flow_fall=None,
                x_offset=None, x_au=None, roi_boxes=None):
        B = x_flow_rise.shape[0]
        if roi_boxes is None or roi_boxes.shape != (B, self.n_roi, 4):
            raise ValueError('ROI-token model requires roi_boxes [B,n_roi,4] from preprocessing')
        geometry = None
        if self.geometry_conditioning != 'none':
            with torch.no_grad():
                _, geometry = self.face_landback(F.interpolate(x_apex, size=112))
        # Encode full-face onset ONCE, shared by the two phases.
        appearance = (self.appearance_encoder(x_onset)
                      if self.appearance_encoder is not None else None)
        au_embed = None
        if self.use_au:
            if x_au is None:
                raise ValueError('AU-enabled model requires x_au')
            au_embed = self.au_encoder(x_au)
        rise = self.motion_encoder.forward_token_rois(x_flow_rise)
        aux = {'rise': None, 'fall': None, 'agreement': None}
        if not self.use_rise_fall or (not self.training and self.rise_fall_mode == 'decision_level'):
            out, feat, attn = self._classify_rois(rise, geometry, appearance, roi_boxes, au_embed)
            return out, feat, attn, aux
        if x_flow_fall is None:
            raise ValueError('Fall motion required during dual-phase training / feature_gate eval')
        fall = self.motion_encoder.forward_token_rois(x_flow_fall)
        if self.rise_fall_mode == 'feature_gate':
            aux['rise'] = self.aux_head_rise(rise.mean(dim=(1, 3, 4)))
            aux['fall'] = self.aux_head_fall(fall.mean(dim=(1, 3, 4)))
            fused, agreement = self.rise_fall_fusion(rise.flatten(0, 1), fall.flatten(0, 1))
            aux['agreement'] = agreement.reshape(B, self.n_roi, 1, self.roi_pool_size, self.roi_pool_size)
            out, feat, attn = self._classify_rois(fused.reshape_as(rise), geometry, appearance, roi_boxes, au_embed)
        else:
            out, feat, attn = self._classify_rois(rise, geometry, appearance, roi_boxes, au_embed)
            aux['fall'], _, _ = self._classify_rois(fall, geometry, appearance, roi_boxes, au_embed)
        return out, feat, attn, aux
