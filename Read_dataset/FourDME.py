import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

EMOTION2IDX = {
    "Negative": 0,
    "Surprise": 1,
    "Positive": 2,
    "Others":   3,
}
IDX2EMOTION = {v: k for k, v in EMOTION2IDX.items()}


class FourDME_Dataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None,
                 stats_path=None, verbose=False):

        self.transform = transform
        split = "train" if is_train else "test"
        self.data_dir = os.path.join(root_dir, split)

        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"Missing directory: {self.data_dir}")

        # load normalization stats 
        self.stats = None
        if stats_path and os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.stats = {k: torch.from_numpy(v).float() for k, v in stats.items()}

        # FLAME keys from SMIRK
        self.keys = ["exp", "jaw", "eyelid", "pose", "shape"]

        # scan dataset 
        self.samples = []
        skipped = 0


        for folder in sorted(os.listdir(self.data_dir)):
            path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(path):
                continue

            img_path = os.path.join(path, "inputs.png")
            label_path = os.path.join(path, "label.npy")

            if not os.path.exists(img_path):
                skipped += 1
                continue
            if not os.path.exists(label_path):
                skipped += 1
                continue

            # check full set of npy files
            npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]
            if len(npy_files) < len(self.keys):
                skipped += 1
                continue

            self.samples.append({
                "folder":   path,
                "img_path": img_path,
            })

        if verbose:
            print(f"[4DME] split={split} | samples={len(self.samples)} | skipped={skipped}")
            # print class distribution
            labels = [int(np.load(os.path.join(s["folder"], "label.npy"))) for s in self.samples]
            for idx, name in IDX2EMOTION.items():
                print(f"  {name}: {labels.count(idx)}")

    def __len__(self):
        return len(self.samples)

    def _load_npy(self, folder, key):
        path = os.path.join(folder, f"{key}.npy")
        if not os.path.exists(path):
            return None
        x = torch.from_numpy(np.load(path)).float()
        if self.stats and f"{key}_mean" in self.stats:
            x = (x - self.stats[f"{key}_mean"]) / self.stats[f"{key}_std"]
        return x

    def __getitem__(self, idx):
        s = self.samples[idx]

        # image 
        img = Image.open(s["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # npy features 
        features = {k: self._load_npy(s["folder"], k) for k in self.keys}
        features = {k: v for k, v in features.items() if v is not None}

        label = int(np.load(os.path.join(s["folder"], "label.npy"))) 

        return {
            "image": img,
            "npy":   features,
            "label": torch.tensor(label, dtype=torch.long),
        }


def compute_4dme_stats(root_dir, split="train", output="4dme_stats.npz"):
    split_dir = os.path.join(root_dir, split)
    keys = ["exp", "jaw", "eyelid", "pose", "shape"]
    buffers = defaultdict(list)
    used = skipped = 0

    for folder in os.listdir(split_dir):
        path = os.path.join(split_dir, folder)
        if not os.path.isdir(path):
            continue

        npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]
        if len(npy_files) < len(keys):
            skipped += 1
            continue

        for key in keys:
            fpath = os.path.join(path, f"{key}.npy")
            if os.path.exists(fpath):
                buffers[key].append(np.load(fpath))

        used += 1

    stats = {}
    for k, v in buffers.items():
        arr  = np.stack(v)
        mean = arr.mean(0)
        std  = arr.std(0)
        std[std < 1e-6] = 1.0
        stats[f"{k}_mean"] = mean
        stats[f"{k}_std"]  = std
        print(f"{k}: shape={mean.shape}")

    out_path = os.path.join(root_dir, output)
    np.savez(out_path, **stats)
    print(f"Saved stats → {out_path}")
    print(f"Used={used} | Skipped={skipped}")


if __name__ == "__main__":

    compute_4dme_stats(
        root_dir="D:/Learning/Lab/MICRO EXPRESSION/MA3D-Net/datasets/4dme_ma3d", 
        split="train", 
        output="4dme_stats.npz"
    )