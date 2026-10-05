"""Synchronized geometric augmentation, aware of motion representation."""
import numpy as np


def flip_sample(rise, fall, boxes, metadata, au=None):
    if metadata.get("box_format") != "normalized_xyxy_exclusive":
        raise ValueError("Expected normalized xyxy ROI metadata")
    R = rise.shape[0]
    if boxes.shape != (R, 4):
        raise ValueError("ROI count differs between boxes and motion")
    permutation = np.asarray(metadata["roi_flip_permutation"], dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(R)):
        raise ValueError("Invalid anatomical ROI reflection permutation")
    if not np.array_equal(permutation[permutation], np.arange(R)):
        raise ValueError("ROI reflection permutation must be an involution")

    def flip_motion(value, phase):
        if value is None:
            return None
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("Geometric augmentation requires spatial [R,3,H,W] maps")
        result = np.flip(value, axis=-1).copy()
        mode = metadata["motion_mode"]
        if mode == "flow":
            normalization = metadata["normalization"]
            if normalization == "minmax":
                # N(-u) = 1 - N(u), except for a constant whole-face channel.
                constant_key = f"u_constant_{phase}"
                if constant_key not in metadata:
                    raise ValueError(f"Missing {constant_key}; regenerate normalization metadata")
                result[:, 0] = 0 if metadata[constant_key] else 1 - result[:, 0]
            elif normalization == "none":
                result[:, 0] *= -1
            else:
                raise ValueError(f"Unsupported normalization: {normalization}")
        elif mode != "pixeldiff":
            raise ValueError(f"Unsupported motion representation: {mode}")
        return result[permutation].copy()

    reflected = boxes.copy()
    reflected[:, 0] = 1 - boxes[:, 2]
    reflected[:, 2] = 1 - boxes[:, 0]
    if au is not None:
        au_perm = np.asarray(metadata["au_flip_permutation"], dtype=np.int64)
        if sorted(au_perm.tolist()) != list(range(len(au))) or not np.array_equal(au_perm[au_perm], np.arange(len(au))):
            raise ValueError("Invalid AU reflection permutation")
        au = au[au_perm].copy()
    return flip_motion(rise, "rise"), flip_motion(fall, "fall"), reflected[permutation].copy(), au
