"""Shared normalized face coordinates; pixel-edge boxes in [0, 1]."""
import torch
from torch import nn
from torch.nn import functional as F


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
    """One attention layer across ROI summaries; no artificial image lattice."""
    def __init__(self, n_roi, channels):
        super().__init__()
        self.identity = nn.Parameter(torch.zeros(1, n_roi, channels))
        nn.init.trunc_normal_(self.identity, std=.02)
        self.position = nn.Linear(4, channels)
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, 4, batch_first=True)

    def forward(self, local, boxes):
        center = (boxes[..., :2] + boxes[..., 2:]) * .5
        size = boxes[..., 2:] - boxes[..., :2]
        tokens = local + self.identity + self.position(torch.cat((center, size), dim=-1))
        normalized = self.norm(tokens)
        exchange, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return (tokens + exchange).mean(dim=1)
