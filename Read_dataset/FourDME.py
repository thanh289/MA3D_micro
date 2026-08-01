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
        inputs.png          -- apex frame, 224x224 RGB
        onset.png            -- onset frame, 224x224 RGB
        offset.png            -- offset frame, 224x224 RGB. Only required
                                 when load_offset=True (motion_backbone=
                                 "rmt" in MA3D.py, which computes its
                                 whole-face pixel-diff directly from RGB
                                 apex/onset/offset rather than from a
                                 pre-cropped ROI flow map).
        flow_map.npy          -- [n_roi, 3, H, W] rise-phase (onset->apex)
                                 optical-flow map (cnn mode) (or flow.npy,
                                 [n_roi*8] pooled vector, legacy 'pool' mode
                                 -- see flow_key)
        flow_map_fall.npy      -- [n_roi, 3, H, W] fall-phase (apex->offset)
                                 optical-flow map, SAME grid as flow_map.npy
                                 (or flow_fall.npy, pooled 'pool' mode --
                                 see flow_fall_key). Only required when
                                 use_rise_fall=True.
        label.npy              -- scalar int64, index into EMOTION2IDX
        fold.npy                -- original 4DME fold id, kept for reference
                                 only, NOT used to decide train/test
                                 membership anymore


    Subject id is parsed from the folder name (see run_inference_flow.py:
    f"{sub_id}_vid{...}_clip{...}_me{...}") and exposed via `self.subjects`,
    a list parallel to `self.samples`, so LOSO (or any other subject-
    grouped) split can be built externally, e.g. with
    sklearn.model_selection.LeaveOneGroupOut.
    """

    def __init__(self, root_dir, transform=None, flow_key="flow_map",
                 flow_fall_key="flow_map_fall", use_rise_fall=True,
                 load_offset=False, load_au=False, file_suffix="", verbose=False):
        """
        flow_key: "flow_map" (spatial map, matches MotionEncoderCNN -- the
            path used by the current architecture) or "flow" (pooled
            vector, legacy 'mlp'-style mode, kept only for comparison).
        flow_fall_key: fall-phase (apex->offset) counterpart of flow_key.
            Only read when use_rise_fall=True.
        use_rise_fall: if True, every sample must have BOTH the rise-phase
            AND fall-phase base flow file on disk to be included (samples
            missing either are skipped, counted in `skipped`) -- keeps
            train/val views the same length for Subset() index alignment
            in get_loso_dataloaders(). If False, flow_fall is never read
            or required (dataset behaves exactly like the original,
            rise-only version).
        load_offset: if True, every sample must ALSO have offset.png on
            disk (skipped otherwise, counted in `skipped`) -- needed for
            MA3D.py's motion_backbone="rmt" path only. False by default
            (offset.png isn't read at all, matching the original
            interface exactly) -- flow_map_fall.npy already covers the
            "fall phase" motion signal for motion_backbone="cnn", so most
            use_rise_fall=True runs do NOT need load_offset=True too.
        load_au: if True, every sample must ALSO have au.npy on disk
            (skipped otherwise, counted in `skipped`) -- a fixed-length
            [40] float32 multi-hot AU vector (20 dynamic + 20 static AUs,
            see au_utils.py) produced by the preprocessing notebook via
            au_utils.save_au_vector(), feeding MA3D.py's optional
            AU-guidance branch (use_au=True). au.npy has NO file_suffix
            variant (it's a ground-truth annotation, not GAMDSS-corrected
            frame data) -- the same file is used regardless of
            file_suffix.
        file_suffix: "" reads the base files (inputs.png, onset.png,
            flow_map.npy, flow_map_fall.npy, offset.png). "_gamdss" (or
            any other suffix produced by run_inference_flow.py --suffix)
            prefers the suffixed variant of EACH file, falling back to the
            base file per-sample, per-file if the suffixed one is missing.
        transform: a PairedFaceTransform-like callable,
            transform(apex_pil, onset_pil, offset_pil=None) ->
            (apex_tensor, onset_tensor) or (apex_tensor, onset_tensor,
            offset_tensor) if offset_pil was passed. A plain
            torchvision.transforms.Compose does NOT work here -- it has no
            notion of jointly transforming multiple images. Its `.train`
            attribute (True/False) also gates whether flow augmentation is
            applied in this dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.flow_key = flow_key
        self.flow_fall_key = flow_fall_key
        self.use_rise_fall = use_rise_fall
        self.load_offset = load_offset
        self.load_au = load_au
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
            base_apex_path   = os.path.join(path, "inputs.png")
            base_onset_path  = os.path.join(path, "onset.png")
            base_offset_path = os.path.join(path, "offset.png")
            base_flow_path   = os.path.join(path, f"{flow_key}.npy")
            base_fall_path   = os.path.join(path, f"{flow_fall_key}.npy")
            base_au_path     = os.path.join(path, "au.npy")
            label_path       = os.path.join(path, "label.npy")

            required_ok = (os.path.exists(base_apex_path) and os.path.exists(base_onset_path)
                           and os.path.exists(base_flow_path) and os.path.exists(label_path))
            if use_rise_fall:
                required_ok = required_ok and os.path.exists(base_fall_path)
            if load_offset:
                required_ok = required_ok and os.path.exists(base_offset_path)
            if load_au:
                required_ok = required_ok and os.path.exists(base_au_path)

            if not required_ok:
                skipped += 1
                continue

            if file_suffix:
                apex_path  = _resolve_with_fallback(
                    path, f"inputs{file_suffix}.png", "inputs.png", fallback_counter)
                onset_path = _resolve_with_fallback(
                    path, f"onset{file_suffix}.png", "onset.png", fallback_counter)
                flow_path  = _resolve_with_fallback(
                    path, f"{flow_key}{file_suffix}.npy", f"{flow_key}.npy", fallback_counter)
                flow_fall_path = None
                if use_rise_fall:
                    flow_fall_path = _resolve_with_fallback(
                        path, f"{flow_fall_key}{file_suffix}.npy",
                        f"{flow_fall_key}.npy", fallback_counter)
                offset_path = None
                if load_offset:
                    offset_path = _resolve_with_fallback(
                        path, f"offset{file_suffix}.png", "offset.png", fallback_counter)
            else:
                apex_path, onset_path, flow_path = base_apex_path, base_onset_path, base_flow_path
                flow_fall_path = base_fall_path if use_rise_fall else None
                offset_path = base_offset_path if load_offset else None

            # au.npy has no file_suffix variant (ground-truth annotation,
            # not GAMDSS-corrected frame data) -- always the base path.
            au_path = base_au_path if load_au else None

            sub_id = folder.split("_vid")[0]

            self.samples.append({
                "folder":         path,
                "apex_path":      apex_path,
                "onset_path":     onset_path,
                "offset_path":    offset_path,
                "flow_path":      flow_path,
                "flow_fall_path": flow_fall_path,
                "au_path":        au_path,
                "label_path":     label_path,
            })
            self.subjects.append(sub_id)

        if verbose:
            print(f"[4DME] samples={len(self.samples)} | skipped={skipped} | "
                  f"unique subjects={len(set(self.subjects))} | "
                  f"use_rise_fall={use_rise_fall} | load_offset={load_offset} | "
                  f"load_au={load_au} | "
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
        offset_img = Image.open(s["offset_path"]).convert("RGB") if s["offset_path"] is not None else None

        if self.transform is not None:
            if offset_img is not None:
                apex_t, onset_t, offset_t = self.transform(apex_img, onset_img, offset_img)
            else:
                apex_t, onset_t = self.transform(apex_img, onset_img)
                offset_t = None
        else:
            # Fallback: bare, un-normalized tensors -- for debugging only.
            apex_t = torch.from_numpy(np.array(apex_img)).permute(2, 0, 1).float() / 255.0
            onset_t = torch.from_numpy(np.array(onset_img)).permute(2, 0, 1).float() / 255.0
            offset_t = (torch.from_numpy(np.array(offset_img)).permute(2, 0, 1).float() / 255.0
                        if offset_img is not None else None)

        flow_rise_np = np.load(s["flow_path"])
        flow_fall_np = np.load(s["flow_fall_path"]) if s["flow_fall_path"] is not None else None
        au_np = np.load(s["au_path"]) if s["au_path"] is not None else None
        label = int(np.load(s["label_path"]))

        if getattr(self.transform, "train", False) and random.random() < 0.5:
            flow_rise_np = np.flip(flow_rise_np, axis=-1).copy()  # flip width axis, every ROI
            flow_rise_np[:, 0, :, :] *= -1                         # u-channel (index 0): flip sign
            if flow_fall_np is not None:
                flow_fall_np = np.flip(flow_fall_np, axis=-1).copy()
                flow_fall_np[:, 0, :, :] *= -1

        item = {
            "apex":      apex_t,
            "onset":     onset_t,
            "flow_rise": torch.from_numpy(flow_rise_np).float(),
            "label":     torch.tensor(label, dtype=torch.long),
        }
        if flow_fall_np is not None:
            item["flow_fall"] = torch.from_numpy(flow_fall_np).float()
        if au_np is not None:
            item["au"] = torch.from_numpy(au_np).float()
        if offset_t is not None:
            item["offset"] = offset_t

        return item