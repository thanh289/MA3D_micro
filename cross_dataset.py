"""Utilities for source-train/validation cross-dataset ME evaluation.

This module intentionally does not touch the legacy ``eval.py`` path, which
belongs to the macro-expression version of the repository.  It builds on the
current motion datasets and MA3D forward signature only.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from Read_dataset.CasmeII import CASME2_Dataset
from Read_dataset.FourDME import FourDME_Dataset
from Read_dataset.SmicHS import SmicHS_Dataset
from engine import prepare_batch
from paired_transform import PairedFaceTransform


CLASS_NAMES = ["Negative", "Positive", "Surprise"]
LABELS = [0, 1, 2]

# Native AU schemas, copied from the preprocessing notebooks.  4DME stores
# [dynamic 18, static/(k) 18]; CASME II merged-side stores one 19-D vector.
FOURDME_AU_VOCAB = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 24, 25, 39, 43, 45]
CASME2_AU_VOCAB = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 16, 17, 18, 20, 24, 25, 26, 38]
COMMON15_AU_VOCAB = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 24, 25]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_cls: type
    display_name: str
    native_au_dim: Optional[int]
    native_au_vocab: Tuple[int, ...]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "4dme": DatasetSpec(
        key="4dme",
        dataset_cls=FourDME_Dataset,
        display_name="4DME",
        native_au_dim=36,
        native_au_vocab=tuple(FOURDME_AU_VOCAB),
    ),
    "casme2": DatasetSpec(
        key="casme2",
        dataset_cls=CASME2_Dataset,
        display_name="CASME II",
        native_au_dim=19,
        native_au_vocab=tuple(CASME2_AU_VOCAB),
    ),
    "smic_hs": DatasetSpec(
        key="smic_hs",
        dataset_cls=SmicHS_Dataset,
        display_name="SMIC-HS",
        native_au_dim=None,
        native_au_vocab=(),
    ),
}

_DATASET_ALIASES = {
    "4dme": "4dme",
    "4dme_motion": "4dme",
    "casme": "casme2",
    "casme2": "casme2",
    "casmeii": "casme2",
    "casme_ii": "casme2",
    "smic": "smic_hs",
    "smic_hs": "smic_hs",
    "smichs": "smic_hs",
}


def normalize_dataset_key(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in _DATASET_ALIASES:
        raise ValueError(
            f"Unknown dataset {value!r}; expected one of 4dme, casme2, smic_hs"
        )
    return _DATASET_ALIASES[key]


def au_dim_for_schema(dataset_key: str, schema: str) -> int:
    dataset_key = normalize_dataset_key(dataset_key)
    spec = DATASET_SPECS[dataset_key]
    if schema == "common15":
        if dataset_key not in ("4dme", "casme2"):
            raise ValueError("AU common15 is defined only for 4DME and CASME II")
        return len(COMMON15_AU_VOCAB)
    if schema == "native":
        if spec.native_au_dim is None:
            raise ValueError(f"{spec.display_name} has no supported GT AU schema")
        return spec.native_au_dim
    raise ValueError(f"Unknown AU schema {schema!r}; expected 'native' or 'common15'")


def validate_au_protocol(source: str, targets: Sequence[str], schema: str) -> int:
    """Validate one-checkpoint AU compatibility and return model ``au_dim``."""
    source = normalize_dataset_key(source)
    targets = [normalize_dataset_key(x) for x in targets]
    keys = [source, *targets]

    unsupported = [k for k in keys if DATASET_SPECS[k].native_au_dim is None]
    if unsupported:
        names = ", ".join(DATASET_SPECS[k].display_name for k in unsupported)
        raise ValueError(
            f"This run uses AU, but {names} has no supported GT AU input. "
            "Run the SMIC target in a separate source model with --use_au disabled."
        )

    if schema == "common15":
        return len(COMMON15_AU_VOCAB)

    dims = {DATASET_SPECS[k].native_au_dim for k in keys}
    vocabs = {DATASET_SPECS[k].native_au_vocab for k in keys}
    if len(dims) != 1 or len(vocabs) != 1:
        details = ", ".join(
            f"{DATASET_SPECS[k].display_name}={DATASET_SPECS[k].native_au_dim}D"
            for k in keys
        )
        raise ValueError(
            f"Native AU schemas are incompatible in one checkpoint ({details}). "
            "Use --au_schema common15 for 4DME <-> CASME II."
        )
    return int(next(iter(dims)))


def project_au_vector(vector: torch.Tensor, dataset_key: str, schema: str) -> torch.Tensor:
    """Project a native dataset AU vector into the selected model schema."""
    dataset_key = normalize_dataset_key(dataset_key)
    expected_native_dim = DATASET_SPECS[dataset_key].native_au_dim
    if expected_native_dim is None:
        raise ValueError(f"{DATASET_SPECS[dataset_key].display_name} has no supported GT AU")
    if vector.ndim != 1 or vector.numel() != expected_native_dim:
        raise ValueError(
            f"{DATASET_SPECS[dataset_key].display_name} AU expected shape "
            f"({expected_native_dim},), got {tuple(vector.shape)}"
        )

    if schema == "native":
        return vector
    if schema != "common15":
        raise ValueError(f"Unknown AU schema: {schema!r}")

    if dataset_key == "4dme":
        # Only the dynamic half [0:18] is eligible.  Static/(k) slots
        # [18:36] are deliberately ignored for cross-dataset alignment.
        native_index = {au: i for i, au in enumerate(FOURDME_AU_VOCAB)}
    elif dataset_key == "casme2":
        native_index = {au: i for i, au in enumerate(CASME2_AU_VOCAB)}
    else:
        raise ValueError("AU common15 is defined only for 4DME and CASME II")

    indices = torch.tensor(
        [native_index[au] for au in COMMON15_AU_VOCAB],
        dtype=torch.long,
        device=vector.device,
    )
    return vector.index_select(0, indices)


class CrossDatasetView(Dataset):
    """Adds sample ids and optional AU projection without changing source files."""

    def __init__(self, base: Dataset, dataset_key: str, use_au: bool, au_schema: str):
        self.base = base
        self.dataset_key = normalize_dataset_key(dataset_key)
        self.use_au = use_au
        self.au_schema = au_schema

    @property
    def samples(self):
        return self.base.samples

    @property
    def subjects(self):
        return self.base.subjects

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = dict(self.base[index])
        if self.use_au:
            if "au" not in item:
                raise RuntimeError(
                    f"Sample {self.sample_id(index)!r} has no AU although use_au=True"
                )
            item["au"] = project_au_vector(
                item["au"], self.dataset_key, self.au_schema
            )
        item["sample_id"] = self.sample_id(index)
        return item

    def sample_id(self, index: int) -> str:
        return os.path.basename(os.path.normpath(self.base.samples[index]["folder"]))


def _native_au_shape_check(base: Dataset, dataset_key: str) -> None:
    expected = DATASET_SPECS[dataset_key].native_au_dim
    if expected is None:
        raise ValueError(f"{DATASET_SPECS[dataset_key].display_name} has no supported GT AU")
    bad: List[str] = []
    for sample in base.samples:
        path = sample.get("au_path")
        shape = tuple(np.load(path, mmap_mode="r").shape) if path else None
        if shape != (expected,):
            bad.append(f"{os.path.basename(sample['folder'])}: {shape}")
            if len(bad) == 5:
                break
    if bad:
        raise ValueError(
            f"Unexpected native AU shape in {DATASET_SPECS[dataset_key].display_name}; "
            f"expected ({expected},). Examples: {bad}"
        )


def _validate_base_dataset(base: Dataset, dataset_key: str, n_roi: int, use_au: bool) -> None:
    if len(base) == 0:
        raise ValueError(
            f"No usable samples found in {base.root_dir!r} for "
            f"{DATASET_SPECS[dataset_key].display_name}"
        )

    labels = [int(np.load(s["label_path"])) for s in base.samples]
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(
            f"Cross-dataset protocol requires labels 0/1/2 only; found {unknown} "
            f"in {base.root_dir!r}"
        )

    bad_flow: List[str] = []
    for sample in base.samples:
        shape = tuple(np.load(sample["flow_path"], mmap_mode="r").shape)
        if len(shape) != 4 or shape[0] != n_roi or shape[1] != 3:
            bad_flow.append(f"{os.path.basename(sample['folder'])}: {shape}")
            if len(bad_flow) == 5:
                break
    if bad_flow:
        raise ValueError(
            f"Expected flow shape [n_roi={n_roi}, 3, H, W]. Examples of mismatch: "
            f"{bad_flow}"
        )

    if use_au:
        _native_au_shape_check(base, dataset_key)


def build_dataset_view(
    dataset_key: str,
    root: str,
    *,
    train: bool,
    n_roi: int,
    use_rise_fall: bool,
    motion_backbone: str,
    use_au: bool,
    au_schema: str,
    use_gamdss: bool = False,
    verbose: bool = False,
) -> CrossDatasetView:
    dataset_key = normalize_dataset_key(dataset_key)
    spec = DATASET_SPECS[dataset_key]
    root = os.path.abspath(os.path.expanduser(root))
    transform = PairedFaceTransform(img_size=224, train=train)
    base = spec.dataset_cls(
        root,
        transform=transform,
        flow_key="flow_map",
        flow_fall_key="flow_map_fall",
        use_rise_fall=use_rise_fall,
        load_offset=motion_backbone == "rmt",
        load_au=use_au,
        file_suffix="_gamdss" if train and use_gamdss else "",
        verbose=verbose,
    )
    _validate_base_dataset(base, dataset_key, n_roi=n_roi, use_au=use_au)
    return CrossDatasetView(base, dataset_key, use_au=use_au, au_schema=au_schema)


def _labels_from_view(view: CrossDatasetView) -> np.ndarray:
    return np.asarray(
        [int(np.load(sample["label_path"])) for sample in view.samples],
        dtype=np.int64,
    )


def subject_group_split(
    subjects: Sequence[str],
    labels: Sequence[int],
    *,
    val_fraction: float,
    seed: int,
    explicit_val_subjects: Optional[Sequence[str]] = None,
    require_all_classes: bool = True,
    max_attempts: int = 500,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    subjects_arr = np.asarray([str(x) for x in subjects])
    labels_arr = np.asarray(labels, dtype=np.int64)
    unique_subjects = sorted(set(subjects_arr.tolist()))
    if len(unique_subjects) < 2:
        raise ValueError("A source train/validation split needs at least two subjects")

    if explicit_val_subjects:
        val_subjects = sorted(set(str(x) for x in explicit_val_subjects))
        missing = sorted(set(val_subjects) - set(unique_subjects))
        if missing:
            raise ValueError(
                f"Unknown --val_subjects {missing}; available subjects: {unique_subjects}"
            )
        val_mask = np.isin(subjects_arr, val_subjects)
        train_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)
        candidates = [(train_idx, val_idx)]
    else:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("--val_fraction must be between 0 and 1")
        candidates = []
        dummy = np.zeros(len(subjects_arr), dtype=np.float32)
        for attempt in range(max_attempts):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=seed + attempt,
            )
            candidates.append(next(splitter.split(dummy, labels_arr, groups=subjects_arr)))

    for train_idx, val_idx in candidates:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        if require_all_classes:
            if set(labels_arr[train_idx].tolist()) != set(LABELS):
                continue
            if set(labels_arr[val_idx].tolist()) != set(LABELS):
                continue
        val_subjects = sorted(set(subjects_arr[val_idx].tolist()))
        return np.asarray(train_idx), np.asarray(val_idx), val_subjects

    coverage = {
        subject: sorted(set(labels_arr[subjects_arr == subject].tolist()))
        for subject in unique_subjects
    }
    raise ValueError(
        "Could not construct a subject-disjoint source split with the required "
        f"class coverage after {len(candidates)} attempt(s). Subject coverage: {coverage}. "
        "Use --allow_missing_val_classes or provide --val_subjects explicitly."
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


def _make_train_sampler(
    labels: np.ndarray, train_idx: np.ndarray, generator: torch.Generator
) -> WeightedRandomSampler:
    split_labels = labels[train_idx].tolist()
    class_counts = Counter(split_labels)
    weights = [1.0 / class_counts[label] for label in split_labels]
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_source_loaders(
    dataset_key: str,
    root: str,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_fraction: float,
    val_subjects: Optional[Sequence[str]],
    require_all_classes: bool,
    use_sampler: bool,
    n_roi: int,
    use_rise_fall: bool,
    motion_backbone: str,
    use_au: bool,
    au_schema: str,
    use_gamdss: bool,
) -> Tuple[DataLoader, DataLoader, Mapping[str, object]]:
    dataset_key = normalize_dataset_key(dataset_key)
    train_view = build_dataset_view(
        dataset_key,
        root,
        train=True,
        n_roi=n_roi,
        use_rise_fall=use_rise_fall,
        motion_backbone=motion_backbone,
        use_au=use_au,
        au_schema=au_schema,
        use_gamdss=use_gamdss,
        verbose=True,
    )
    val_view = build_dataset_view(
        dataset_key,
        root,
        train=False,
        n_roi=n_roi,
        use_rise_fall=use_rise_fall,
        motion_backbone=motion_backbone,
        use_au=use_au,
        au_schema=au_schema,
        use_gamdss=False,
        verbose=False,
    )

    train_ids = [os.path.basename(x["folder"]) for x in train_view.samples]
    val_ids = [os.path.basename(x["folder"]) for x in val_view.samples]
    if train_ids != val_ids:
        raise RuntimeError(
            "Source train/validation dataset views do not contain the same ordered samples"
        )

    labels = _labels_from_view(train_view)
    train_idx, val_idx, held_out_subjects = subject_group_split(
        train_view.subjects,
        labels,
        val_fraction=val_fraction,
        seed=seed,
        explicit_val_subjects=val_subjects,
        require_all_classes=require_all_classes,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    train_subset = Subset(train_view, train_idx.tolist())
    val_subset = Subset(val_view, val_idx.tolist())

    if use_sampler:
        train_loader = DataLoader(
            train_subset,
            sampler=_make_train_sampler(labels, train_idx, generator),
            drop_last=len(train_subset) >= batch_size,
            **loader_common,
        )
    else:
        train_loader = DataLoader(
            train_subset,
            shuffle=True,
            drop_last=len(train_subset) >= batch_size,
            **loader_common,
        )
    val_loader = DataLoader(
        val_subset,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    metadata = {
        "dataset": dataset_key,
        "root": os.path.abspath(os.path.expanduser(root)),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "train_subjects": sorted(set(np.asarray(train_view.subjects)[train_idx].tolist())),
        "val_subjects": held_out_subjects,
        "train_class_counts": dict(Counter(labels[train_idx].tolist())),
        "val_class_counts": dict(Counter(labels[val_idx].tolist())),
    }
    return train_loader, val_loader, metadata


def build_target_loader(
    dataset_key: str,
    root: str,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    n_roi: int,
    use_rise_fall: bool,
    motion_backbone: str,
    use_au: bool,
    au_schema: str,
) -> Tuple[DataLoader, Mapping[str, object]]:
    dataset_key = normalize_dataset_key(dataset_key)
    view = build_dataset_view(
        dataset_key,
        root,
        train=False,
        n_roi=n_roi,
        use_rise_fall=use_rise_fall,
        motion_backbone=motion_backbone,
        use_au=use_au,
        au_schema=au_schema,
        use_gamdss=False,
        verbose=True,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    labels = _labels_from_view(view)
    metadata = {
        "dataset": dataset_key,
        "root": os.path.abspath(os.path.expanduser(root)),
        "samples": len(view),
        "subjects": len(set(view.subjects)),
        "class_counts": dict(Counter(labels.tolist())),
    }
    return loader, metadata


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    per_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "UF1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "UAR": float(recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        # WAR = class recalls weighted by class support, algebraically the
        # same as overall accuracy in this single-label setting.
        "WAR": float(accuracy_score(y_true, y_pred)),
        "Samples": int(len(y_true)),
        "F1-Pos": float(per_f1[1]),
        "F1-Neg": float(per_f1[0]),
        "F1-Surp": float(per_f1[2]),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


@torch.no_grad()
def predict_loader(model, loader: DataLoader, criterion, device: str):
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []
    sample_ids: List[str] = []

    for batch in loader:
        apex, onset, flow_rise, flow_fall, offset, au, labels = prepare_batch(batch, device)
        logits, _, _, _ = model(apex, onset, flow_rise, flow_fall, offset, au)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_logits.append(logits.detach().cpu())
        sample_ids.extend([str(x) for x in batch["sample_id"]])

    if total == 0:
        raise ValueError("Evaluation loader is empty")
    return {
        "loss": total_loss / total,
        "labels": torch.cat(all_labels).numpy(),
        "preds": torch.cat(all_preds).numpy(),
        "logits": torch.cat(all_logits).numpy(),
        "sample_ids": sample_ids,
    }


def save_json(path: str, value: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def save_prediction_csv(path: str, prediction: Mapping[str, object]) -> None:
    logits = np.asarray(prediction["logits"])
    labels = np.asarray(prediction["labels"])
    preds = np.asarray(prediction["preds"])
    sample_ids = list(prediction["sample_ids"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "label",
                "label_name",
                "prediction",
                "prediction_name",
                "logit_negative",
                "logit_positive",
                "logit_surprise",
            ]
        )
        for sample_id, label, pred, row in zip(sample_ids, labels, preds, logits):
            writer.writerow(
                [
                    sample_id,
                    int(label),
                    CLASS_NAMES[int(label)],
                    int(pred),
                    CLASS_NAMES[int(pred)],
                    *[float(x) for x in row],
                ]
            )


def write_result_table(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    columns = [
        "Source",
        "Target",
        "UF1",
        "UAR",
        "WAR",
        "Samples",
        "F1-Pos",
        "F1-Neg",
        "F1-Surp",
        "BestEpoch",
        "Seed",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
