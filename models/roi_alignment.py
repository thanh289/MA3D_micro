"""Shared normalized face coordinates; pixel-edge boxes in [0, 1]."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def sample_rois(features, boxes, output_size):
    """Sample the same anatomical boxes on RGB or feature maps.

    boxes: [B,R,4] (x0,y0,x1,y1), exclusive upper pixel edges,
    normalized by image width/height. Returns [B*R,C,h,w].
    Feature-map sampling is region alignment, not exact receptive-field
    correspondence. Border values are replicated for coarse feature maps.
    """
    B, R, _ = boxes.shape
    if features.shape[0] != B or not torch.isfinite(boxes).all():
        raise ValueError("Invalid ROI boxes or batch size")
    if ((boxes < 0) | (boxes > 1)).any() or (boxes[..., 2:] <= boxes[..., :2]).any():
        raise ValueError("ROI boxes must be nondegenerate normalized xyxy")
    h, w = output_size
    boxes = boxes.to(device=features.device, dtype=features.dtype)
    x = (torch.arange(w, device=features.device, dtype=features.dtype) + .5) / w
    y = (torch.arange(h, device=features.device, dtype=features.dtype) + .5) / h
    gx = boxes[..., 0, None] + x * (boxes[..., 2] - boxes[..., 0])[..., None]
    gy = boxes[..., 1, None] + y * (boxes[..., 3] - boxes[..., 1])[..., None]
    grid = torch.stack((gx[..., None, :].expand(B, R, h, w),
                        gy[..., :, None].expand(B, R, h, w)), dim=-1)
    sources = features[:, None].expand(B, R, *features.shape[1:]).reshape(B * R, *features.shape[1:])
    return F.grid_sample(sources, (2 * grid - 1).reshape(B * R, h, w, 2),
                         mode="bilinear", padding_mode="border", align_corners=False)


class ROIContextAggregator(nn.Module):
    """Small transformer over ROI summaries, independent of appearance fusion."""
    def __init__(self, n_roi, channels, depth=2, heads=4, dropout=0.1,
                 use_position=True):
        super().__init__()
        if depth not in (1, 2) or channels % heads:
            raise ValueError('Use 1-2 ROI blocks and channels divisible by heads')
        self.identity = nn.Parameter(torch.zeros(1, n_roi, channels))
        nn.init.trunc_normal_(self.identity, std=.02)
        self.position = nn.Linear(4, channels) if use_position else None
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(channels, heads, dim_feedforward=2 * channels,
                                       dropout=dropout, activation='gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(channels)

    def forward(self, local, boxes):
        center = (boxes[..., :2] + boxes[..., 2:]) * .5
        size = boxes[..., 2:] - boxes[..., :2]
        tokens = local + self.identity
        if self.position is not None:
            tokens = tokens + self.position(torch.cat((center, size), dim=-1))
        with sdpa_kernel(SDPBackend.MATH):
            for block in self.blocks:
                tokens = block(tokens)
        return self.norm(tokens).mean(dim=1)


class LearnedROIPooling(nn.Module):
    """Shared spatial scorer; softmax is applied inside each ROI separately."""
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Sequential(nn.Conv2d(channels, channels // 4, 1), nn.GELU(),
                                   nn.Conv2d(channels // 4, 1, 1))

    def forward(self, local):
        B, R, C, H, W = local.shape
        weights = self.score(local.reshape(B * R, C, H, W)).flatten(2).softmax(-1)
        weights = weights.reshape(B, R, 1, H, W)
        return (local * weights).sum(dim=(-1, -2)), weights


class ROIGeometryResidual(nn.Module):
    """One same-box modulation block with a small, learned residual scale.

    Sampling the coarse geometry map aligns regions, not physical pixels.
    There is no cross-ROI mixing here; ROI interaction happens afterwards.
    """
    def __init__(self, channels, scale_init=0.01):
        super().__init__()
        self.condition = nn.Sequential(nn.Conv2d(512, channels, 1), nn.GELU(),
                                       nn.Conv2d(channels, 2 * channels, 1))
        self.norm = nn.GroupNorm(1, channels)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, motion, geometry, boxes):
        B, R, C, H, W = motion.shape
        context = sample_rois(geometry, boxes, (H, W))
        gamma, beta = self.condition(context).chunk(2, dim=1)
        local = motion.reshape(B * R, C, H, W)
        delta = torch.tanh(gamma) * self.norm(local) + torch.tanh(beta)
        return (local + self.scale * delta).reshape_as(motion)
