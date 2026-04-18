import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class RAFDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None,
                 stats_path="rafdb_emoca_stats.npz", verbose=False):

        self.transform = transform
        split = "train" if is_train else "test"
        self.data_dir = os.path.join(root_dir, split)

        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"Missing directory: {self.data_dir}")

        # ---- load normalization stats ----
        self.stats = None
        stats_file = os.path.join(root_dir, stats_path)
        if os.path.exists(stats_file):
            self.stats = {
                k: torch.from_numpy(v).float()
                for k, v in np.load(stats_file).items()
            }

        # ---- dataset config ----
        self.keys = ["shape", "tex", "exp", "pose", "detail"]
        self.shape_sizes = {"shape": 100, "tex": 50, "exp": 50, "pose": 6, "detail": 128}

        # ---- build index ----
        self.samples = []
        skipped = 0

        for label_dir in sorted(os.listdir(self.data_dir)):
            label_path = os.path.join(self.data_dir, label_dir)
            if not os.path.isdir(label_path):
                continue

            label = int(label_dir) - 1

            for sample_dir in os.listdir(label_path):
                path = os.path.join(label_path, sample_dir)
                if not os.path.isdir(path):
                    continue

                img_path = os.path.join(path, f"{sample_dir}.jpg")

                # require image + enough npy files
                npy_files = [f for f in os.listdir(path)
                             if f.endswith(".npy") and not f.endswith("_lmk.npy")]

                if not os.path.exists(img_path) or len(npy_files) < 5:
                    skipped += 1
                    continue

                self.samples.append({"folder": path, "img": img_path, "label": label})

        if verbose:
            print(f"[RAFDataset] samples={len(self.samples)} skipped={skipped}")

    def __len__(self):
        return len(self.samples)

    def _load_npy(self, folder, key):
        path = os.path.join(folder, f"{key}.npy")

        if os.path.exists(path):
            x = torch.from_numpy(np.load(path)).float()
        else:
            x = torch.zeros(self.shape_sizes[key])

        # normalize if stats exist
        if self.stats and f"{key}_mean" in self.stats:
            x = (x - self.stats[f"{key}_mean"]) / self.stats[f"{key}_std"]

        return x

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---- image ----
        img = Image.open(sample["img"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # ---- features ----
        features = {k: self._load_npy(sample["folder"], k) for k in self.keys}

        return {
            "image": img,
            "npy": features,
            "label": torch.tensor(sample["label"])
        }


# =========================================================
# Compute dataset statistics
# =========================================================
def compute_stats(root_dir, split="train",
                  output_name="rafdb_emoca_stats.npz",
                  skip_landmark=True,
                  min_samples=10):

    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        raise RuntimeError(f"Missing split: {split_dir}")

    buffers = defaultdict(list)

    print(f"[Stats] scanning {split}")

    for label_dir in os.listdir(split_dir):
        label_path = os.path.join(split_dir, label_dir)
        if not os.path.isdir(label_path):
            continue

        for sample_dir in os.listdir(label_path):
            path = os.path.join(label_path, sample_dir)
            if not os.path.isdir(path):
                continue

            npy_files = [f for f in os.listdir(path) if f.endswith(".npy")]

            valid = False
            for f in npy_files:
                if skip_landmark and f.endswith("_lmk.npy"):
                    continue

                key = f[:-4]
                x = np.load(os.path.join(path, f)).reshape(-1)

                buffers[key].append(x)
                valid = True

            if not valid:
                continue

    # ---- compute stats ----
    stats = {}

    for key, values in buffers.items():
        if len(values) < min_samples:
            continue

        data = np.stack(values)
        mean, std = data.mean(0), data.std(0)
        std = np.where(std < 1e-6, 1.0, std)

        stats[f"{key}_mean"] = mean
        stats[f"{key}_std"] = std

        print(f"{key}: {data.shape}")

    out_path = os.path.join(root_dir, output_name)
    np.savez(out_path, **stats)

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    compute_stats(
        root_dir="Datasets/RAF-DB_lmk",
        split="train",
        output_name="rafdb_emoca_stats.npz"
    )