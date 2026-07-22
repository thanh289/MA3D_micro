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

    def __init__(self, root_dir, transform=None, flow_key="flow_map", verbose=False):
        """
        flow_key: "flow_map" (spatial map, matches MotionEncoderCNN -- the
            path used by the current architecture) or "flow" (pooled
            vector, legacy 'mlp'-style mode, kept only for comparison).
        transform: a PairedFaceTransform-like callable,
            transform(apex_pil, onset_pil) -> (apex_tensor, onset_tensor).
            A plain torchvision.transforms.Compose does NOT work here --
            it has no notion of jointly transforming 2 images. Its `.train`
            attribute (True/False) also gates whether flow augmentation is
            applied in this dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.flow_key = flow_key
        self.flow_filename = f"{flow_key}.npy"

        if not os.path.exists(root_dir):
            raise RuntimeError(f"Missing directory: {root_dir}")

        self.samples = []
        self.subjects = []
        skipped = 0

        for folder in sorted(os.listdir(root_dir)):
            path = os.path.join(root_dir, folder)
            if not os.path.isdir(path):
                continue

            apex_path  = os.path.join(path, "inputs.png")
            onset_path = os.path.join(path, "onset.png")
            flow_path  = os.path.join(path, self.flow_filename)
            label_path = os.path.join(path, "label.npy")

            if not (os.path.exists(apex_path) and os.path.exists(onset_path)
                    and os.path.exists(flow_path) and os.path.exists(label_path)):
                skipped += 1
                continue

            # Subject id = the part of the folder name before "_vid".
            # NOTE: assumes SubID never itself contains the literal
            # substring "_vid" -- true for typical short subject codes,
            # but worth a quick sanity check if your SubID format is
            # unusual (e.g. contains "_vid" as part of a longer code).
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
                  f"unique subjects={len(set(self.subjects))}")
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
            # Fallback: bare, un-normalized tensors -- for debugging only.
            apex_t = torch.from_numpy(np.array(apex_img)).permute(2, 0, 1).float() / 255.0
            onset_t = torch.from_numpy(np.array(onset_img)).permute(2, 0, 1).float() / 255.0

        flow_np = np.load(s["flow_path"])
        label = int(np.load(s["label_path"]))

        # Augment on the NUMPY array, before any torch conversion (see
        # class docstring for why order matters here).
        if getattr(self.transform, "train", False) and random.random() < 0.5:
            flow_np = np.flip(flow_np, axis=-1).copy()  # flip width axis, every ROI
            flow_np[:, 0, :, :] *= -1                    # u-channel (index 0): flip sign

        flow = torch.from_numpy(flow_np).float()

        return {
            "apex":  apex_t,
            "onset": onset_t,
            "flow":  flow,
            "label": torch.tensor(label, dtype=torch.long),
        }