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