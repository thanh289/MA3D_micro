import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkPriorEncoder(nn.Module):
    """
    Pools the dense landmark feature map (from the frozen MobileFaceNet
    backbone) into a single embedding vector, used as the FiLM condition
    source that modulates the motion branch.

    This replaces the old SMIRK/3DMM-parameter encoder entirely. The
    landmark tokens now play the role the 3DMM parameters used to play (a
    compact descriptor of WHERE things are on the face), while carrying
    essentially no motion information themselves -- exactly the property
    wanted from a modulator source, as opposed to the motion branch which
    is now the one that needs to carry the "what changed" content.

    Input:  x_lmk [B, N, in_dim]  (N=49 landmark tokens, in_dim=512, from
            MobileFaceNet's conv_features reshaped the same way as before)
    Output: [B, embed_dim]
    """

    def __init__(self, in_dim=512, embed_dim=256, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x_lmk):
        pooled = x_lmk.mean(dim=1)  # [B, in_dim] -- simple mean pool over the 49 tokens
        return self.net(pooled)


class FiLMConditionGenerator(nn.Module):
    """
    Renamed from ThreeDMMConditionGenerator -- logic unchanged. Turns a
    single condition embedding into a (gamma, beta) pair per FiLM block.
    """

    def __init__(self, embed_dim, num_blocks, channels):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_blocks * channels * 2)
        )

    def forward(self, embed):
        B = embed.shape[0]

        gamma_beta = self.net(embed)
        gamma_beta = gamma_beta.view(
            B, self.num_blocks, 2, self.channels
        )

        gamma = gamma_beta[:, :, 0]
        beta = gamma_beta[:, :, 1]

        return gamma, beta


class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.net(x)
        return x * scale


class FiLMAdaptiveBlock(nn.Module):
    """Renamed from ThreeDMMAdaptiveBlock -- logic unchanged (FiLM modulation
    + bottleneck conv + optional SE, with a residual connection)."""

    def __init__(self, channels, bottleneck_ratio=4, use_se=False, dropout=0.1):
        super().__init__()
        self.use_se = use_se
        hidden = channels // bottleneck_ratio

        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, channels, 1)

        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(hidden, affine=False)

        self.dropout = nn.Dropout2d(dropout)

        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # FiLM modulation
        out = self.norm1(x)
        out = out * (1 + gamma) + beta
        out = F.leaky_relu(out, 0.2)

        # Bottleneck
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.leaky_relu(out, 0.2)

        out = self.conv2(out)
        out = self.dropout(out)

        out = F.leaky_relu(out, 0.2)
        out = self.conv3(out)

        if self.use_se:
            out = self.se(out)

        return x + out


class LandmarkModulationFusion(nn.Module):
    """
    Renamed from ThreeDMMFusion. Modulates the motion feature map using a
    condition vector derived from the (frozen) landmark branch, via a chain
    of FiLM blocks.

    Bug fix vs. the old ThreeDMMFusion: the old class took a `num_blocks`
    argument and stored it in `self.num_blocks`, but the actual
    `nn.ModuleList` was hardcoded to exactly 5 entries regardless of the
    value passed in -- if `num_blocks != 5` was ever used, `film_generator`
    would produce a mismatched number of (gamma, beta) pairs and
    `gamma[:, i]` would eventually index out of range. The block list is
    now built dynamically from `num_blocks`.
    """

    def __init__(self, feat_nc=512, embed_dim=256, num_blocks=5,
                 landmark_dim=512, landmark_hidden_dim=256):
        super().__init__()

        self.num_blocks = num_blocks

        self.landmark_encoder = LandmarkPriorEncoder(
            in_dim=landmark_dim,
            embed_dim=embed_dim,
            hidden_dim=landmark_hidden_dim,
        )

        self.film_generator = FiLMConditionGenerator(
            embed_dim,
            num_blocks,
            feat_nc,
        )

        self.blocks = nn.ModuleList([
            FiLMAdaptiveBlock(feat_nc, use_se=True) for _ in range(num_blocks)
        ])

    def forward(self, x_motion, x_lmk):
        cond_embed = self.landmark_encoder(x_lmk)
        gamma, beta = self.film_generator(cond_embed)

        x = x_motion
        for i, block in enumerate(self.blocks):
            x = block(x, gamma[:, i], beta[:, i])

        return x


# =============================================================================
# Route B (decided in chat): SPATIAL FiLM.
#
# The classes above implement FiLM the standard way -- (gamma, beta) is a
# per-CHANNEL vector [B, C], pooled from the landmark map's 49 tokens down
# to a single global descriptor, then broadcast UNIFORMLY over the entire
# 7x7 motion feature map. That matches the ORIGINAL MA3D-Net's rationale
# (3DMM params are a global identity/shape descriptor with no natural
# spatial axis), but it directly contradicts what the landmark branch is
# supposed to do in the ME architecture: "tell the motion branch which
# AU-region to trust/amplify" is a claim about WHERE on the face, which a
# channel-only, spatially-uniform gamma/beta cannot express at all, no
# matter how well it's trained.
#
# x_lmk_map (the landmark branch's raw [B, 512, 7, 7] conv_features, BEFORE
# it gets flattened into 49 tokens) already sits on the exact same 7x7 grid
# as x_motion -- both come from a 224x224 input through a stride-32
# backbone -- so no resize/pooling is needed at all: position (i,j) in the
# landmark map is the same facial region as position (i,j) in the motion
# map. The classes below use that alignment directly to produce a
# PER-POSITION (gamma, beta), i.e. an actual spatial gate/reliability map
# in the spirit of SAC^2-Net's reliability-aware fusion, instead of a
# single global scale/shift.
# =============================================================================


class SpatialFiLMConditionGenerator(nn.Module):
    """
    Spatial counterpart of FiLMConditionGenerator. Keeps the landmark map's
    spatial structure and produces a PER-POSITION (gamma, beta) pair for
    every FiLM block, via 1x1 convs (a 1x1 conv is a per-position linear
    layer, applied identically at all 49 grid locations -- the direct
    spatial analogue of the original's per-sample MLP).

    One shared trunk generates conditions for ALL num_blocks blocks at
    once (same synchronization rationale as the original
    FiLMConditionGenerator: all blocks stay driven by the same underlying
    landmark evidence, instead of each block interpreting it independently).

    Input:  x_lmk_map [B, in_dim, H, W]  (H=W=7 in this codebase)
    Output: gamma, beta -- each [B, num_blocks, channels, H, W]
    """

    def __init__(self, in_dim, num_blocks, channels, hidden_dim=128):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        self.trunk = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.InstanceNorm2d(hidden_dim, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Conv2d(hidden_dim, num_blocks * channels * 2, kernel_size=1)

    def forward(self, x_lmk_map):
        h = self.trunk(x_lmk_map)
        gamma_beta = self.head(h)  # [B, num_blocks*channels*2, H, W]

        B, _, H, W = gamma_beta.shape
        gamma_beta = gamma_beta.view(B, self.num_blocks, 2, self.channels, H, W)

        gamma = gamma_beta[:, :, 0]  # [B, num_blocks, channels, H, W]
        beta = gamma_beta[:, :, 1]
        return gamma, beta


class SpatialFiLMAdaptiveBlock(nn.Module):
    """
    Same bottleneck-conv + optional SE + residual structure as
    FiLMAdaptiveBlock. The only difference is forward() takes gamma/beta
    already shaped [B, C, H, W] (spatial) and applies them directly, with
    no unsqueeze/broadcast step -- every internal conv here uses
    padding=1, stride=1 (no spatial downsampling anywhere in this block),
    so a 7x7 gamma/beta stays aligned with x at every stage.
    """

    def __init__(self, channels, bottleneck_ratio=4, use_se=False, dropout=0.1):
        super().__init__()
        self.use_se = use_se
        hidden = channels // bottleneck_ratio

        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, channels, 1)

        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(hidden, affine=False)

        self.dropout = nn.Dropout2d(dropout)

        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x, gamma, beta):
        # gamma, beta: [B, C, H, W] -- already spatial, applied directly
        out = self.norm1(x)
        out = out * (1 + gamma) + beta
        out = F.leaky_relu(out, 0.2)

        out = self.conv1(out)
        out = self.norm2(out)
        out = F.leaky_relu(out, 0.2)

        out = self.conv2(out)
        out = self.dropout(out)

        out = F.leaky_relu(out, 0.2)
        out = self.conv3(out)

        if self.use_se:
            out = self.se(out)

        return x + out


class SpatialLandmarkModulationFusion(nn.Module):
    """
    Spatial (Route B) replacement for LandmarkModulationFusion. Takes
    x_lmk_map DIRECTLY -- the [B, 512, 7, 7] conv_features from the frozen
    MobileFaceNet backbone, BEFORE any flattening into 49 tokens -- since
    pooling to a vector is exactly the step that threw away the spatial
    specificity this module needs. No LandmarkPriorEncoder in this path.
    """

    def __init__(self, feat_nc=512, landmark_dim=512, num_blocks=5,
                 hidden_dim=128):
        super().__init__()
        self.num_blocks = num_blocks

        self.condition_generator = SpatialFiLMConditionGenerator(
            in_dim=landmark_dim,
            num_blocks=num_blocks,
            channels=feat_nc,
            hidden_dim=hidden_dim,
        )

        self.blocks = nn.ModuleList([
            SpatialFiLMAdaptiveBlock(feat_nc, use_se=True) for _ in range(num_blocks)
        ])

    def forward(self, x_motion, x_lmk_map):
        gamma, beta = self.condition_generator(x_lmk_map)  # each [B, num_blocks, C, H, W]

        x = x_motion
        for i, block in enumerate(self.blocks):
            x = block(x, gamma[:, i], beta[:, i])

        return x