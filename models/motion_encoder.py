import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoderCNN(nn.Module):
    """
    Motion branch encoder -- the main "content" branch in the new
    architecture (replaces the old X_face/IR50-on-apex role). Trained from
    scratch (no ImageNet/ArcFace pretraining makes sense here: the input is
    an optical-flow map, not a natural face image, so a pretrained RGB-face
    backbone would mostly see a domain it never learned).

    Design rationale (Hướng B, decided in chat):
      - Keeps the "process each ROI separately, weight-shared" idea that
        empirically beat the single merged-composite design in isolated
        testing (2 ROI -> MEAN-style composite was worse; 3 ROI with
        separate-ROI conv beat MEAN once an eye ROI was added).
      - BUT does not fully global-average-pool each ROI down to a single
        vector (which is what the original ThreeDMMEncoderCNN prior-encoder
        did, since its role was only to feed a FiLM modulator). Here the
        output MUST retain spatial structure, since this branch is now the
        one participating in cross-attention with the appearance branch.
      - Each ROI keeps a small spatial grid (roi_pool_size x roi_pool_size)
        instead of collapsing to 1x1, the three ROI grids are laid out
        side-by-side into one canvas, and a final AdaptiveAvgPool2d forces
        the output to exactly 7x7 regardless of input resolution -- this
        keeps the output shape compatible with the existing feat_nc=512,
        7x7 interface used by LandmarkModulationFusion / pyramid_fuse
        without touching hyp_crossvit.py at all.

    Input:  x_roi [B, n_roi, 3, H, W]  (channels: u, v, optical-strain;
            matches flow_map.npy produced by run_inference_flow.py with
            --feature_mode cnn, n_roi=3 -> eyebrow, eye, mouth)
    Output: [B, out_channels, 7, 7]
    """

    def __init__(self, n_roi=3, out_channels=512, roi_pool_size=3, dropout=0.2):
        super().__init__()
        self.n_roi = n_roi
        self.roi_pool_size = roi_pool_size

        # Weight-shared across ROI (n_roi folded into the batch dimension in
        # forward()) -- matches the design that performed well in the
        # standalone CNN test mentioned in chat.
        self.roi_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),                                   # e.g. 28 -> 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((roi_pool_size, roi_pool_size)),  # NOT (1,1) -- keep spatial
            nn.Dropout2d(dropout),
        )

        self.merge = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((7, 7)),  # force exactly 7x7, independent of n_roi/roi_pool_size
        )

    def forward(self, x_roi):
        B, R, C, H, W = x_roi.shape
        assert R == self.n_roi, f"expected n_roi={self.n_roi}, got {R}"

        x = x_roi.view(B * R, C, H, W)
        feat = self.roi_conv(x)  # [B*R, 64, roi_pool_size, roi_pool_size]

        feat = feat.view(B, R, 64, self.roi_pool_size, self.roi_pool_size)
        # lay the R ROI grids side by side along width -> [B, 64, roi_pool_size, roi_pool_size*R]
        feat = feat.permute(0, 2, 3, 1, 4).reshape(
            B, 64, self.roi_pool_size, self.roi_pool_size * R
        )

        return self.merge(feat)  # [B, out_channels, 7, 7]