"""ROI motion queries full-face geometry; no crop of the geometry feature map."""
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


def roi_token_coordinates(boxes, grid_size):
    """Normalized face coordinates of uniform ROI-bin centers, ROI-major order."""
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError('roi_boxes must be [B,R,4] normalized xyxy')
    if not torch.isfinite(boxes).all() or ((boxes < 0) | (boxes > 1)).any():
        raise ValueError('roi_boxes must be finite and within [0,1]')
    if (boxes[..., 2:] <= boxes[..., :2]).any():
        raise ValueError('roi_boxes must have positive width and height')
    centers = (torch.arange(grid_size, device=boxes.device, dtype=boxes.dtype) + .5) / grid_size
    yy, xx = torch.meshgrid(centers, centers, indexing='ij')
    local = torch.stack((xx, yy), dim=-1).reshape(1, 1, -1, 2)
    xy = boxes[..., None, :2] + local * (boxes[..., None, 2:] - boxes[..., None, :2])
    return xy.flatten(1, 2)


class FullFaceGeometryContext(nn.Module):
    def __init__(self, channels=512, num_heads=4, use_position=True):
        super().__init__()
        if num_heads <= 0 or channels % num_heads:
            raise ValueError('geometry_heads must be a positive divisor of 512')
        self.use_position = use_position
        self.position = nn.Linear(2, channels) if use_position else None
        self.motion_norm = nn.LayerNorm(channels)
        self.geometry_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def positions(self, xy):
        return self.position(xy) if self.position is not None else 0

    def forward(self, motion, geometry, xy):
        # Q indexes motion; K/V index the full face. Output follows motion order.
        B, C, H, W = geometry.shape
        yy, xx = torch.meshgrid(
            (torch.arange(H, device=geometry.device, dtype=geometry.dtype) + .5) / H,
            (torch.arange(W, device=geometry.device, dtype=geometry.dtype) + .5) / W,
            indexing='ij')
        face_xy = torch.stack((xx, yy), dim=-1).reshape(1, H * W, 2)
        face = geometry.flatten(2).transpose(1, 2)
        query = self.motion_norm(motion) + self.positions(xy)
        key = self.geometry_norm(face) + self.positions(face_xy)
        with sdpa_kernel(SDPBackend.MATH):
            context, _ = self.attention(query, key, face, need_weights=False)
        return context


def add_roi_token_args(parser):
    """Shared CLI for LOSO and cross-dataset training; old defaults untouched."""
    parser.add_argument('--roi_pool_size', type=int, choices=[2, 3, 4, 5, 7], default=3,
                        help='g x g tokens per motion ROI; no preprocessing change needed.')
    parser.add_argument('--geometry_heads', type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument('--geometry_conditioning', choices=['cross_attention', 'global_mean', 'none'],
                        default='cross_attention', help='Geometry-to-FiLM ablation.')
    parser.add_argument('--appearance_fusion', choices=['cross_attention', 'none'],
                        default='cross_attention',
                        help='none removes onset encoder and pyramid; mean-pools motion after FiLM.')
    parser.add_argument('--no_roi_position', action='store_true',
                        help='Disable explicit coordinate encoding (learned pyramid positions remain).')


def roi_token_kwargs(args):
    return dict(roi_pool_size=args.roi_pool_size, geometry_heads=args.geometry_heads,
                geometry_conditioning=args.geometry_conditioning,
                use_roi_position=not args.no_roi_position,
                appearance_fusion=args.appearance_fusion)
