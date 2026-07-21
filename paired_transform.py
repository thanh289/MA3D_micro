import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PairedFaceTransform:
    """
    Transform applied jointly to an (apex, onset) image pair.

    Rationale: apex and onset are 2 frames of the SAME clip, a few dozen
    milliseconds apart, so they realistically share near-identical
    lighting/color conditions. Calling a plain torchvision.transforms
    pipeline independently on each frame (as the old single-image
    train_transform did) would let ColorJitter draw two DIFFERENT random
    color shifts for the two frames -- injecting a "different lighting"
    cue between apex/onset that never occurs in real data. This class
    draws the ColorJitter parameters ONCE per sample and applies them
    identically to both frames instead.

    RandomErasing is applied independently per frame: it's a small-area
    (5%) synthetic occlusion, there's no physical reason it should hit the
    same spot on both frames, and decorrelating it is a reasonable
    regularizer on its own.

    The actual ME motion signal (the flow map) never goes through this
    transform at all -- it's precomputed offline by run_inference_flow.py
    and loaded as a raw array, so none of this augmentation touches it.
    """

    def __init__(self, img_size=224, train=True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 jitter_brightness=0.2, jitter_contrast=0.2, jitter_saturation=0.2,
                 erase_scale=(0.05, 0.05)):
        self.img_size = img_size
        self.train = train
        self.normalize = T.Normalize(mean=mean, std=std)

        if train:
            self.jitter_brightness = jitter_brightness
            self.jitter_contrast = jitter_contrast
            self.jitter_saturation = jitter_saturation
            self.random_erasing = T.RandomErasing(p=1, scale=erase_scale)

    def __call__(self, apex_img, onset_img):
        apex_img = TF.resize(apex_img, [self.img_size, self.img_size])
        onset_img = TF.resize(onset_img, [self.img_size, self.img_size])

        if self.train:
            fn_idx, b, c, s, h = T.ColorJitter.get_params(
                (max(0.0, 1 - self.jitter_brightness), 1 + self.jitter_brightness),
                (max(0.0, 1 - self.jitter_contrast), 1 + self.jitter_contrast),
                (max(0.0, 1 - self.jitter_saturation), 1 + self.jitter_saturation),
                None,  # no hue jitter, matches the old ColorJitter(0.2, 0.2, 0.2) call
            )
            apex_img = self._apply_jitter(apex_img, fn_idx, b, c, s, h)
            onset_img = self._apply_jitter(onset_img, fn_idx, b, c, s, h)

        apex_t = TF.to_tensor(apex_img)
        onset_t = TF.to_tensor(onset_img)
        apex_t = self.normalize(apex_t)
        onset_t = self.normalize(onset_t)

        if self.train:
            apex_t = self.random_erasing(apex_t)
            onset_t = self.random_erasing(onset_t)

        return apex_t, onset_t

    @staticmethod
    def _apply_jitter(img, fn_idx, brightness_factor, contrast_factor,
                       saturation_factor, hue_factor):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = TF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = TF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = TF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = TF.adjust_hue(img, hue_factor)
        return img