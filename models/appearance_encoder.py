import torch
import torch.nn as nn
import torch.nn.functional as F


class _Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _Attention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Block(nn.Module):
    """Standard pre-norm Transformer block (Attention + MLP), same shape as
    the ones used in PC_vit.py, just trimmed of drop_path/timm dependencies
    to keep this file self-contained (matches the style of motion_encoder.py
    / ThreeDMM_Adaptive.py -- pure torch.nn, no external framework)."""

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AppearanceEncoderViT(nn.Module):
    """
    Appearance-context branch (replaces the frozen IR50-on-onset branch).

    Design rationale (Route C1s):
      - The old branch used a FULL-DEPTH, FROZEN IR50 checkpoint trained
        under an ArcFace objective -- whose explicit goal is to become
        INVARIANT to expression (keep identity, discard everything else)
        so it can tell people apart regardless of what face they're
        making. Using that same backbone, frozen, to supply "where on the
        face is this happening" context is working against its own
        training objective -- risk is larger here than in the MaE-domain
        original MA3D-Net, since ME motion is already a much weaker signal
        to begin with.
      - TSFmicro and GAMDSS both use a deliberately SHALLOW, TRAINED
        (not frozen, not identity-pretrained) transformer for the
        equivalent "static/context" branch, specifically to avoid
        re-learning identity while still extracting positional structure.
      - This class borrows GAMDSS's actual "vit_pos" design verbatim in
        spirit (see PC_vit.py / model.py::cov_DRT): the raw RGB frame is
        bilinearly downsampled DIRECTLY to a tiny grid (no conv stem
        first), patchified with patch_size=1 (a per-remaining-pixel linear
        projection), then passed through a shallow (depth=2) Transformer
        trained from scratch. GAMDSS then ADDS this into their main
        branch's bottleneck feature; here we instead feed it as the
        cross-attention partner into the EXISTING pyramid_fuse module
        (unchanged), since Huong C1 as scoped only swaps the encoder, not
        the fusion mechanism.
      - grid_size=7 is chosen (not GAMDSS's 14) specifically so the output
        token count (49) matches pyramid_fuse's in_chans=49/q_chanel=49
        without any change to hyp_crossvit.py.

    Input:  x [B, 3, 224, 224] RGB (e.g. onset frame, already normalized)
    Output: [B, 49, 512] tokens
    """

    def __init__(self, grid_size=7, embed_dim=512, depth=2, num_heads=4,
                 mlp_ratio=4., drop=0.1, attn_drop=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size * grid_size

        # patch_size=1 conv over the ALREADY-DOWNSAMPLED grid == a per-pixel
        # linear projection from 3 channels to embed_dim (matches PC_vit's
        # PatchEmbed with patch_size=1 applied to a pre-resized image).
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=1, stride=1)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            _Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        # x: [B, 3, H, W] raw RGB -> bilinearly downsample directly to the
        # tiny grid (no conv stem) -- intentional: keeps this branch coarse
        # ("where" structure only), matching GAMDSS's vit_pos exactly.
        x = F.interpolate(x, size=(self.grid_size, self.grid_size),
                           mode='bilinear', align_corners=False)
        x = self.patch_embed(x)                # [B, embed_dim, grid, grid]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # [B, grid*grid, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x  # [B, 49, 512]