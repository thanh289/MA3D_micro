import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

EMOTION2IDX = {
    "Negative":   0,
    "Positive":   1,
    "Surprise":   2,
    "Repression": 3,
    "Others":     4,
}
IDX2EMOTION = {v: k for k, v in EMOTION2IDX.items()}


def _resolve_with_fallback(folder, suffixed_name, base_name, fallback_counter):
    """Prefer the suffixed file (e.g. inputs_gamdss.png); if it doesn't
    exist for this sample, fall back to the base file (inputs.png) and bump
    fallback_counter[0]. The base file is assumed to always exist (checked
    by the caller before this is used)"""
    suffixed_path = os.path.join(folder, suffixed_name)
    if os.path.exists(suffixed_path):
        return suffixed_path
    fallback_counter[0] += 1
    return os.path.join(folder, base_name)


class FourDME_Dataset(Dataset):
    """
    Reads samples produced by the updated run_inference_flow.py: a single
    flat directory (root_dir) of sample folders -- no train/ or test/
    subfolder anymore, since train/test membership is decided at LOSO-split
    time (see build_dataloader.py::get_loso_dataloaders), not baked into
    preprocessing.

    Each sample folder is expected to contain:
        inputs.png     -- apex frame, 224x224 RGB
        onset.png      -- onset frame, 224x224 RGB
        flow_map.npy   -- [n_roi, 3, H, W] optical-flow map (cnn mode)
                           (or flow.npy, [n_roi*8] pooled vector, legacy
                           'pool' mode -- see flow_key)
        label.npy       -- scalar int64, index into EMOTION2IDX
        fold.npy         -- original 4DME fold id, kept for reference only,
                           NOT used to decide train/test membership anymore

    Subject id is parsed from the folder name (see run_inference_flow.py:
    f"{sub_id}_vid{...}_clip{...}_me{...}") and exposed via `self.subjects`,
    a list parallel to `self.samples`, so LOSO (or any other subject-
    grouped) split can be built externally, e.g. with
    sklearn.model_selection.LeaveOneGroupOut.
    """

    def __init__(self, root_dir, transform=None, flow_key="flow_map",
                 file_suffix="", verbose=False):
        """
        flow_key: "flow_map" (spatial map) or "flow" (pooled vector).
        file_suffix: "" reads the base files (inputs.png, onset.png,
            flow_map.npy). "_gamdss" (or any other suffix produced by
            run_inference_flow.py --suffix) prefers inputs_gamdss.png /
            onset_gamdss.png / flow_map_gamdss.npy, falling back to the
            base file per-sample if the suffixed one is missing.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.flow_key = flow_key
        self.file_suffix = file_suffix

        if not os.path.exists(root_dir):
            raise RuntimeError(f"Missing directory: {root_dir}")

        self.samples = []
        self.subjects = []
        skipped = 0
        fallback_counter = [0]  # mutable int, shared across _resolve_with_fallback calls

        for folder in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, folder)
            if not os.path.isdir(path):
                continue

            # Base files MUST exist -- these decide whether the sample is
            # usable at all, independent of file_suffix.
            base_apex_path  = os.path.join(path, "inputs.png")
            base_onset_path = os.path.join(path, "onset.png")
            base_flow_path  = os.path.join(path, f"{flow_key}.npy")
            label_path      = os.path.join(path, "label.npy")

            if not (os.path.exists(base_apex_path) and os.path.exists(base_onset_path)
                    and os.path.exists(base_flow_path) and os.path.exists(label_path)):
                skipped += 1
                continue

            if file_suffix:
                apex_path  = _resolve_with_fallback(
                    path, f"inputs{file_suffix}.png", "inputs.png", fallback_counter)
                onset_path = _resolve_with_fallback(
                    path, f"onset{file_suffix}.png", "onset.png", fallback_counter)
                flow_path  = _resolve_with_fallback(
                    path, f"{flow_key}{file_suffix}.npy", f"{flow_key}.npy", fallback_counter)
            else:
                apex_path, onset_path, flow_path = base_apex_path, base_onset_path, base_flow_path

            sub_id = folder.split("_vid")[0]

            self.samples.append({
                "folder":     path,
                "apex_path":  apex_path,
                "onset_path": onset_path,
                "flow_path":  flow_path,
                "label_path": label_path,
            })
            self.subjects.append(sub_id)

        if verbose:
            print(f"[4DME] samples={len(self.samples)} | skipped={skipped} | "
                  f"unique subjects={len(set(self.subjects))} | "
                  f"file_suffix={file_suffix!r} | fallback_to_base={fallback_counter[0]}")
            labels = [int(np.load(s["label_path"])) for s in self.samples]
            for idx, name in IDX2EMOTION.items():
                print(f"  {name}: {labels.count(idx)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        apex_img  = Image.open(s["apex_path"]).convert("RGB")
        onset_img = Image.open(s["onset_path"]).convert("RGB")

        if self.transform is not None:
            apex_t, onset_t = self.transform(apex_img, onset_img)
        else:
            apex_t = torch.from_numpy(np.array(apex_img)).permute(2, 0, 1).float() / 255.0
            onset_t = torch.from_numpy(np.array(onset_img)).permute(2, 0, 1).float() / 255.0

        flow_np = np.load(s["flow_path"])
        label = int(np.load(s["label_path"]))

        if getattr(self.transform, "train", False) and random.random() < 0.5:
            flow_np = np.flip(flow_np, axis=-1).copy()
            flow_np[:, 0, :, :] *= -1

        flow = torch.from_numpy(flow_np).float()

        return {
            "apex":  apex_t,
            "onset": onset_t,
            "flow":  flow,
            "label": torch.tensor(label, dtype=torch.long),
        }