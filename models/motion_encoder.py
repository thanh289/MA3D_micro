import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicAdaptiveAvgPool2d(nn.Module):
    """
    Drop-in replacement for nn.AdaptiveAvgPool2d, deterministic on CUDA.

    adaptive_avg_pool2d's CUDA backward has NO deterministic implementation
    in PyTorch whenever the input/output spatial ratio isn't evenly
    divisible (see the UserWarning torch.use_deterministic_algorithms(True,
    warn_only=True) raises for it) -- true for BOTH poolings in this file
    (14->3 and (3, 3*n_roi)->(7,7), neither divides evenly). CPU's backward
    for this op IS deterministic, and the tensors going through this layer
    are tiny (64 channels, <=14x14) by this point in the network, so
    round-tripping to CPU just for this op is negligible overhead compared
    to the rest of a training step -- much cheaper than the alternative of
    reworking the pooling into a fixed-kernel op that would change its
    numeric behavior (uneven windows can't be expressed as a single
    fixed-stride nn.AvgPool2d).

    Set `enabled=False` (or leave the module's `.enabled` off) to skip the
    CPU round-trip and fall back to plain GPU adaptive pooling when you
    don't need bit-exact reproducibility (e.g. normal training runs where
    a little non-determinism from this one op is an acceptable tradeoff
    for not paying the CPU<->GPU sync cost every forward/backward).
    """

    def __init__(self, output_size, enabled=True):
        super().__init__()
        self.output_size = output_size
        self.enabled = enabled

    def forward(self, x):
        if self.enabled and x.device.type == "cuda":
            return F.adaptive_avg_pool2d(x.cpu(), self.output_size).to(x.device)
        return F.adaptive_avg_pool2d(x, self.output_size)


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

    def __init__(self, n_roi=3, out_channels=512, roi_pool_size=3, dropout=0.2,
                 deterministic_pool=False):
        """
        deterministic_pool: if True, both AdaptiveAvgPool2d layers below
        round-trip through CPU for their backward pass (see
        DeterministicAdaptiveAvgPool2d's docstring) -- use this when you
        need bit-exact reproducibility across runs (paired with
        torch.use_deterministic_algorithms(True) in train.py). Leave False
        (default) for normal training, where the small non-determinism
        from these 2 ops is an acceptable tradeoff for not paying the
        CPU<->GPU sync cost on every forward/backward.
        """
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
            DeterministicAdaptiveAvgPool2d((roi_pool_size, roi_pool_size),
                                            enabled=deterministic_pool),  # NOT (1,1) -- keep spatial
            nn.Dropout2d(dropout),
        )

        self.merge = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            DeterministicAdaptiveAvgPool2d((7, 7), enabled=deterministic_pool),  # force exactly 7x7, independent of n_roi/roi_pool_size
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