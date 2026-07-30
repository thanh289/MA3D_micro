import torch
import torch.nn as nn
import torch.nn.functional as F

from .RMT import VisRetNet


class RMTMotionEncoder(nn.Module):
    """
    Alternative motion backbone (motion_backbone="rmt" in MA3D.py), swapping
    in GAMDSS's actual RMT_T3/VisRetNet retention-based backbone in place of
    MotionEncoderCNN. RMT.py is copied VERBATIM from the real GAMDSS source
    (only requires `pip install timm` for DropPath/trunc_normal_/
    register_model/_cfg) -- nothing in that file is modified.

    IMPORTANT CAVEAT (read before assuming this matches GAMDSS's real
    behaviour): GAMDSS's own VisRetNet.forward_features() does NOT stop at
    a spatial feature map -- it goes on to add the vit_pos position-
    calibration signal (`x = x + y`), project, BatchNorm, GLOBAL AVERAGE
    POOL, and classify via a linear head, ALL INSIDE VisRetNet itself (see
    RMT.py lines ~579-599). GAMDSS's BDRT.forward() calls
    `out = self.main_branch(act, POS)` and gets FINAL LOGITS directly --
    there is no separate downstream FiLM/cross-attention stage in the real
    architecture at all.

    This class deliberately does NOT replicate that. It STOPS right after
    the 4 retention stages (`patch_embed` + `self.layers`), returning the
    raw spatial feature map [B, 512, 7, 7] instead -- skipping VisRetNet's
    own POS-add / proj / norm / pool / head entirely. The rationale: this
    codebase already has its OWN landmark-based FiLM modulation
    (landmark_fusion in MA3D.py, sourced from the frozen MobileFaceNet
    branch, not vit_pos) and its own classifier (pyramid_fuse + head).
    Injecting GAMDSS's vit_pos-based POS-add INSIDE this encoder as well
    would duplicate/conflate two different "where on the face" signals
    with no clear way to disentangle their separate contributions later.
    So: motion_backbone="rmt" is scoped as "swap ONLY the motion feature
    extractor for a retention-based one", NOT "port GAMDSS's whole
    architecture" -- the latter would mean bypassing landmark_fusion /
    pyramid_fuse / appearance_encoder entirely, a much larger change not
    attempted here.

    Input resolution note: GAMDSS feeds a 224x224 whole-face diff into
    VisRetNet, which downsamples it by /16 (patch_embed's stride-2 conv,
    then 3 PatchMerging stages between the 4 retention stages) to reach a
    14x14 spatial resolution right before the POS-add (POS itself comes
    from vit_pos at 14x14). To land on 7x7 instead (matching this
    codebase's existing landmark_fusion / pyramid_fuse interface, both
    hardcoded for a 7x7 / 49-token grid), this class resizes the INPUT
    down to 112x112 before feeding VisRetNet's patch_embed, rather than
    modifying VisRetNet's downsampling factor itself (112 / 16 = 7). No
    change to the borrowed RMT.py code is needed for this.

    Input:  whole-face pixel-diff, [B, 3, 224, 224] (e.g. apex_224 -
            onset_224 for the rise phase; NOT ROI-cropped -- GAMDSS's real
            architecture has no ROI concept at all, see chat notes)
    Output: [B, 512, 7, 7] (channel count matches embed_dims[-1]=512 in
            RMT_T3's config, so no extra projection layer is needed to
            match the existing feat_nc=512 interface)
    """

    def __init__(self, in_chans=3, input_resize=112):
        super().__init__()
        self.input_resize = input_resize

        # Exact RMT_T3 config from RMT.py's `def RMT_T3(num_class)` factory
        # -- copied here (rather than calling that factory) so we can pass
        # num_classes=0 and never construct/touch the classification tail
        # (proj/norm/swish/avgpool/head) at all, since forward() below
        # bypasses VisRetNet.forward()/forward_features() entirely.
        self.backbone = VisRetNet(
            in_chans=in_chans,
            num_classes=0,
            embed_dims=[64, 128, 256, 512],
            depths=[1, 1, 2, 1],
            num_heads=[2, 2, 2, 2],
            init_values=[2, 2, 2, 2],
            heads_ranges=[4, 4, 6, 6],
            mlp_ratios=[3, 3, 3, 3],
            drop_path_rate=0.1,
            chunkwise_recurrents=[True, True, False, False],
            layerscales=[False, False, False, False],
        )

    def forward(self, x_diff):
        """
        x_diff: [B, 3, H, W] whole-face pixel-diff (any input resolution --
                resized to self.input_resize x self.input_resize first).
        Returns: [B, 512, 7, 7]
        """
        x = F.interpolate(x_diff, size=(self.input_resize, self.input_resize),
                           mode='bilinear', align_corners=False)

        x = self.backbone.patch_embed(x)          # (B, H, W, C), NOT (B,C,H,W) -- RMT's own convention
        for layer in self.backbone.layers:
            x = layer(x)                            # 4 retention stages, 3 PatchMerging downsamples in between

        x = x.permute(0, 3, 1, 2).contiguous()      # (B, H, W, C) -> (B, C, H, W), match MotionEncoderCNN's convention
        return x  # [B, 512, 7, 7] (112 / 16 = 7)