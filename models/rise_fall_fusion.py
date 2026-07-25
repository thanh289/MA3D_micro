import torch
import torch.nn as nn
import torch.nn.functional as F


class RiseFallAgreementFusion(nn.Module):
    """
    Combines the rise-phase (onset->apex) and fall-phase (apex->offset)
    motion features into a single fused motion representation, using a
    per-position "agreement" map as a spatial reliability gate -- the
    ME-domain analogue of CausalNet's core observation: at a genuine AU
    location, the muscle CONTRACTS during rise and RELAXES during fall, so
    the two phases' motion signal points in near-OPPOSITE directions at
    that exact spatial position. A position where rise and fall motion
    instead look SIMILAR (or unrelated) is more likely camera/lighting
    noise, a rigid head movement, or an unrelated facial tic than a real AU.

    DESIGN CAVEAT (read before trusting this blindly): agreement here is
    computed on the ENCODED motion features (X_rise, X_fall, each
    [B,512,7,7], AFTER MotionEncoderCNN), not on the raw (u,v) flow
    channels. This is cheaper (reuses the existing shared-weight
    MotionEncoderCNN call as-is) but conv+ReLU layers are NOT guaranteed to
    preserve the raw flow's "opposite input -> opposite feature vector"
    property -- MotionEncoderCNN was never trained with an objective that
    encourages this. If ablation shows this fusion doesn't help (or hurts),
    the fallback is a CausalNet-faithful version computing agreement on the
    RAW (u,v) angle/direction channels BEFORE the deep encoder (a small
    separate "direction encoder" over flow_map[:, :2] at its native
    28x28-per-ROI resolution) -- not implemented here, flagged for v2.

    agreement(i,j) = (1 - cos_sim(X_rise[:,:,i,j], X_fall[:,:,i,j])) / 2
        -> 0 when the two phases' feature vectors point the SAME way
           (suspicious -- no real contraction/relaxation asymmetry)
        -> 1 when they point OPPOSITE ways (matches the AU prior)
        -> 0.5 when orthogonal / unrelated

    Fusion: X_fused = ((X_rise + X_fall) / 2) * (1 + agreement)
        A simple additive base fusion (average of the two phases -- cheap,
        no extra learned parameters, appropriate given the small dataset),
        spatially reweighted by the agreement map so AU-consistent
        positions get amplified relative to inconsistent ones.

    IMPORTANT: this output REPLACES the old single-phase x_motion as the
    input to the (UNCHANGED) landmark FiLM modulation -- i.e. rise/fall
    fusion and landmark FiLM are STACKED (landmark still gets the final
    say on top of the fused rise+fall representation), not landmark being
    replaced. Matches the "combine, don't replace" decision from chat.

    Input:  x_rise, x_fall -- each [B, C, H, W] (C=512, H=W=7 in this
            codebase), both produced by the SAME MotionEncoderCNN instance
            (shared weights) called on flow_rise / flow_fall respectively.
    Output: x_fused [B, C, H, W] (same shape), agreement [B, 1, H, W]
            (returned too, mainly for inspection/visualization -- e.g.
            plotting where the model considers the AU to genuinely be).
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x_rise, x_fall):
        cos_sim = F.cosine_similarity(x_rise, x_fall, dim=1, eps=self.eps)  # [B, H, W]
        agreement = (1.0 - cos_sim) / 2.0                                   # [B, H, W], in [0, 1]
        agreement = agreement.unsqueeze(1)                                  # [B, 1, H, W]

        x_fused = (x_rise + x_fall) * 0.5
        x_fused = x_fused * (1.0 + agreement)  # broadcasts over channel dim

        return x_fused, agreement