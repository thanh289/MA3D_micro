"""Motion-first ROI baseline and an optional same-region geometry residual."""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F

from .motion_encoder import MotionEncoderCNN
from .mobilefacenet import MobileFaceNet
from .roi_alignment import LearnedROIPooling, ROIContextAggregator, ROIGeometryResidual
from .fullface_context import roi_token_coordinates


class CompactROIModel(nn.Module):
    def __init__(self, *, num_classes, n_roi, motion_dim, roi_pool_size,
                 roi_depth, roi_heads, roi_dropout, use_roi_position,
                 geometry_conditioning, geometry_scale_init, landmark_checkpoint,
                 use_rise_fall, rise_fall_mode):
        super().__init__()
        if motion_dim not in (128, 192):
            raise ValueError('motion_dim must be 128 or 192')
        if n_roi < 1 or num_classes < 2 or roi_pool_size not in (2, 3, 4, 5, 7):
            raise ValueError('Invalid ROI count, class count or ROI grid size')
        if roi_heads < 1 or motion_dim % roi_heads or roi_depth not in (1, 2):
            raise ValueError('Use 1-2 ROI blocks and heads dividing motion_dim')
        if not 0 <= roi_dropout < 1 or not math.isfinite(geometry_scale_init):
            raise ValueError('Invalid dropout or geometry scale')
        if geometry_conditioning not in ('none', 'roi_residual'):
            raise ValueError('roi_transformer supports geometry none or roi_residual')
        if use_rise_fall and rise_fall_mode != 'decision_level':
            raise ValueError('roi_transformer uses fall auxiliary only: select '
                             '--rise_fall_mode decision_level. Direction fusion is not implemented.')
        self.architecture = 'motion_roi_transformer_v1'
        self.n_roi = n_roi
        self.num_classes = num_classes
        self.roi_pool_size = roi_pool_size
        self.use_rise_fall = use_rise_fall
        self.rise_fall_mode = rise_fall_mode
        self.geometry_conditioning = geometry_conditioning
        self.appearance_encoder = None
        self.pyramid_fuse = None
        self.face_landback = None
        self.geometry_residual = None
        self.motion_encoder = MotionEncoderCNN(n_roi=n_roi, out_channels=motion_dim,
                                               roi_pool_size=roi_pool_size,
                                               deterministic_pool=True)
        self.roi_pool = LearnedROIPooling(motion_dim)
        self.roi_aggregator = ROIContextAggregator(n_roi, motion_dim, roi_depth,
                                                  roi_heads, roi_dropout, use_roi_position)
        self.head = nn.Linear(motion_dim, num_classes)
        if geometry_conditioning == 'roi_residual':
            self.face_landback = MobileFaceNet([112, 112], 136)
            if landmark_checkpoint is not None:
                path = Path(landmark_checkpoint)
                if not path.is_file():
                    raise FileNotFoundError(f'Geometry requires MobileFaceNet weights: {path}. '
                                            'Use --geometry_conditioning none for motion-only.')
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                self.face_landback.load_state_dict(checkpoint['state_dict'])
            # None is reserved for restoring the backbone from a complete MA3D checkpoint.
            self.face_landback.requires_grad_(False)
            self.face_landback.eval()
            self.geometry_residual = ROIGeometryResidual(motion_dim, geometry_scale_init)

    def train(self, mode=True):
        super().train(mode)
        if self.face_landback is not None:
            self.face_landback.eval()
        return self

    def _classify_motion(self, motion, geometry, boxes):
        if self.geometry_residual is not None:
            motion = self.geometry_residual(motion, geometry, boxes)
        summaries, weights = self.roi_pool(motion)
        feature = self.roi_aggregator(summaries, boxes)
        return self.head(feature), feature, {'roi_pool_weights': weights}

    def forward(self, x_apex, x_onset, x_flow_rise, x_flow_fall=None,
                x_offset=None, x_au=None, roi_boxes=None):
        if x_flow_rise.ndim != 5 or x_flow_rise.shape[1:3] != (self.n_roi, 3):
            raise ValueError('Expected motion [B,n_roi,3,H,W]')
        B = x_flow_rise.shape[0]
        if roi_boxes is None or roi_boxes.shape != (B, self.n_roi, 4):
            raise ValueError('Expected roi_boxes [B,n_roi,4]')
        boxes = roi_boxes.to(device=x_flow_rise.device, dtype=x_flow_rise.dtype)
        roi_token_coordinates(boxes, 1)  # Validate normalized nondegenerate boxes.
        geometry = None
        if self.face_landback is not None:
            if x_apex is None:
                raise ValueError('Geometry requires the apex RGB frame')
            with torch.no_grad():
                _, geometry = self.face_landback(F.interpolate(x_apex, size=(112, 112),
                                                               mode='bilinear', align_corners=False))
        rise = self.motion_encoder.forward_token_rois(x_flow_rise)
        logits, feature, attention = self._classify_motion(rise, geometry, boxes)
        aux = {'rise': None, 'fall': None, 'agreement': None}
        if self.training and self.use_rise_fall:
            if x_flow_fall is None or x_flow_fall.shape != x_flow_rise.shape:
                raise ValueError('Fall auxiliary training requires matching rise/fall maps')
            fall = self.motion_encoder.forward_token_rois(x_flow_fall)
            aux['fall'], _, _ = self._classify_motion(fall, geometry, boxes)
        return logits, feature, attention, aux
