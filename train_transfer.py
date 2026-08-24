"""Train MA3D on a source train/validation split, then evaluate targets once.

The best checkpoint is selected exclusively by source-validation UF1.  Target
datasets are not loaded until source training is complete, which makes it hard
to accidentally select an epoch from target performance.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from cross_dataset import (
    CLASS_NAMES,
    COMMON15_AU_VOCAB,
    DATASET_SPECS,
    build_source_loaders,
    build_target_loader,
    compute_metrics,
    normalize_dataset_key,
    predict_loader,
    save_json,
    save_prediction_csv,
    validate_au_protocol,
    write_result_table,
)
from engine import train_one_epoch, validate
from loss_function.loss import LabelSmoothingCrossEntropy, MarginAwareCELoss
from models.MA3D import MA3D


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "MA3D source train/validation -> unseen cross-dataset evaluation"
    )

    # Data protocol
    parser.add_argument("--source", required=True, help="4dme or casme2")
    parser.add_argument("--source_root", required=True)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="DATASET=ROOT",
        help="Repeat for each unseen target, e.g. --target casme2=/path/to/data",
    )
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument(
        "--val_subjects",
        default=None,
        help="Optional comma-separated source subject ids; overrides --val_fraction",
    )
    parser.add_argument(
        "--allow_missing_val_classes",
        action="store_true",
        help="Allow a source validation split that does not contain all 3 classes",
    )

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_gamma", type=float, default=0.98)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument("--early_stop_on_perfect_val", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # MA3D architecture/input signature
    parser.add_argument("--model_type", choices=["small", "base", "large"], default="small")
    parser.add_argument("--n_roi", type=int, default=6)
    parser.add_argument(
        "--roi_mode",
        choices=["merged3", "split6", "custom"],
        default=None,
        help="Preprocessing metadata; inferred from n_roi=3/6 when omitted",
    )
    parser.add_argument(
        "--motion_input",
        choices=["flow", "pixeldiff"],
        default="flow",
        help="Preprocessing metadata; both variants use flow_map.npy on disk",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        default=None,
        help="Optional preprocessing crop scale recorded in the checkpoint/W&B contract",
    )
    parser.add_argument(
        "--normalization_tag",
        default="global_minmax",
        help="Preprocessing normalization label recorded for reproducibility",
    )
    parser.add_argument("--landmark_embed_dim", type=int, default=256)
    parser.add_argument("--num_film_blocks", type=int, default=5)
    parser.add_argument("--use_rise_fall", action="store_true")
    parser.add_argument(
        "--rise_fall_mode",
        choices=["feature_gate", "decision_level"],
        default="feature_gate",
    )
    parser.add_argument("--aux_loss_weight", type=float, default=0.2)
    parser.add_argument("--motion_backbone", choices=["cnn", "rmt"], default="cnn")
    parser.add_argument("--use_gamdss", action="store_true")

    # AU: common15 is the supported cross-dataset 4DME/CASME schema.
    parser.add_argument("--use_au", action="store_true")
    parser.add_argument("--au_schema", choices=["common15", "native"], default="common15")
    parser.add_argument("--au_embed_dim", type=int, default=128)

    # Output/resume
    parser.add_argument("--output_dir", default="transfer_runs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Path to a last_checkpoint.pth produced by this script",
    )

    # W&B
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="MA3D-cross-dataset-3class")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_watch_model", action="store_true")
    return parser.parse_args(argv)


def parse_targets(values: Sequence[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --target {value!r}; expected DATASET=ROOT, e.g. casme2=/data/casme"
            )
        raw_key, raw_root = value.split("=", 1)
        key = normalize_dataset_key(raw_key)
        root = raw_root.strip()
        if not root:
            raise ValueError(f"Target root is empty in {value!r}")
        if key in seen:
            raise ValueError(f"Target dataset {key!r} was provided more than once")
        seen.add(key)
        parsed.append((key, os.path.abspath(os.path.expanduser(root))))
    return parsed


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return value


def _load_checkpoint(path: str, device: str) -> Mapping[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced
        return torch.load(path, map_location=device)


def _source_metrics(labels, preds) -> Dict[str, float]:
    metrics = compute_metrics(labels, preds)
    return {
        "UF1": float(metrics["UF1"]),
        "UAR": float(metrics["UAR"]),
        "WAR": float(metrics["WAR"]),
    }


def _plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1, 2], CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks([0, 1, 2], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    threshold = cm.max() / 2 if cm.max() else 0.5
    for row in range(3):
        for column in range(3):
            ax.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
                color="white" if cm[row, column] > threshold else "black",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def _build_model_config(args: argparse.Namespace, au_dim: int) -> Dict[str, object]:
    return {
        "num_classes": 3,
        "type": args.model_type,
        "n_roi": args.n_roi,
        "landmark_embed_dim": args.landmark_embed_dim,
        "num_film_blocks": args.num_film_blocks,
        "use_rise_fall": args.use_rise_fall,
        "rise_fall_mode": args.rise_fall_mode,
        "motion_backbone": args.motion_backbone,
        "use_au": args.use_au,
        "au_dim": au_dim,
        "au_embed_dim": args.au_embed_dim,
    }


def _validate_resume(checkpoint: Mapping[str, object], current: Mapping[str, object]) -> None:
    saved = checkpoint.get("model_config")
    if saved != current:
        raise ValueError(
            "Resume checkpoint model_config does not match the current command.\n"
            f"saved={saved}\ncurrent={current}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = get_args(argv)
    if args.roi_mode is None:
        args.roi_mode = {3: "merged3", 6: "split6"}.get(args.n_roi, "custom")
    expected_roi_count = {"merged3": 3, "split6": 6}.get(args.roi_mode)
    if expected_roi_count is not None and args.n_roi != expected_roi_count:
        raise ValueError(
            f"--roi_mode {args.roi_mode} requires --n_roi {expected_roi_count}, "
            f"got {args.n_roi}"
        )
    source = normalize_dataset_key(args.source)
    if source == "smic_hs":
        raise ValueError("SMIC-HS is target-only in this protocol; choose 4dme or casme2 as source")
    targets = parse_targets(args.target)
    target_keys = [key for key, _ in targets]
    if source in target_keys:
        raise ValueError("Source dataset must not also be listed as an unseen target")

    au_dim = 0
    if args.use_au:
        au_dim = validate_au_protocol(source, target_keys, args.au_schema)
    elif args.au_schema == "native":
        # Schema is irrelevant when AU is disabled; normalize it in saved config.
        args.au_schema = "common15"

    source_root = os.path.abspath(os.path.expanduser(args.source_root))
    device = _resolve_device(args.device)
    set_seed(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_run_name = f"src_{source}_seed{args.seed}_{timestamp}"
    if args.resume_checkpoint and args.run_name is None:
        run_dir = os.path.abspath(os.path.dirname(args.resume_checkpoint))
        run_name = _safe_name(os.path.basename(run_dir))
    else:
        run_name = _safe_name(args.run_name or default_run_name)
        run_dir = os.path.abspath(os.path.join(args.output_dir, run_name))
    best_path = os.path.join(run_dir, "best_checkpoint.pth")
    last_path = os.path.join(run_dir, "last_checkpoint.pth")

    if os.path.exists(best_path) and not args.resume_checkpoint:
        raise FileExistsError(
            f"Run directory already contains {best_path!r}. Use a new --run_name or "
            "pass --resume_checkpoint; existing results are not overwritten implicitly."
        )
    os.makedirs(run_dir, exist_ok=True)

    val_subjects = None
    if args.val_subjects:
        val_subjects = [x.strip() for x in args.val_subjects.split(",") if x.strip()]

    model_config = _build_model_config(args, au_dim=au_dim)
    input_config = {
        "num_classes": 3,
        "class_names": CLASS_NAMES,
        "n_roi": args.n_roi,
        "roi_mode": args.roi_mode,
        "motion_input": args.motion_input,
        "crop_scale": args.crop_scale,
        "normalization_tag": args.normalization_tag,
        "use_rise_fall": args.use_rise_fall,
        "rise_fall_mode": args.rise_fall_mode,
        "motion_backbone": args.motion_backbone,
        "use_gamdss_source_train_only": args.use_gamdss,
        "use_au": args.use_au,
        "au_schema": args.au_schema if args.use_au else None,
        "au_dim": au_dim if args.use_au else None,
        "common_au_vocab": COMMON15_AU_VOCAB if args.use_au and args.au_schema == "common15" else None,
    }
    command_config = {
        **vars(args),
        "source": source,
        "source_root": source_root,
        "targets": {key: root for key, root in targets},
        "run_dir": run_dir,
        "device_resolved": device,
        "model_config": model_config,
        "input_config": input_config,
    }
    save_json(os.path.join(run_dir, "run_config.json"), command_config)

    print(f"Device: {device}")
    print(f"Source: {DATASET_SPECS[source].display_name} -> {source_root}")
    print(
        "Targets: "
        + ", ".join(f"{DATASET_SPECS[key].display_name} -> {root}" for key, root in targets)
    )
    if args.use_au:
        print(f"AU: schema={args.au_schema}, au_dim={au_dim}")
    else:
        print("AU: disabled")

    train_loader, val_loader, split_metadata = build_source_loaders(
        source,
        source_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_fraction=args.val_fraction,
        val_subjects=val_subjects,
        require_all_classes=not args.allow_missing_val_classes,
        use_sampler=args.use_sampler,
        n_roi=args.n_roi,
        use_rise_fall=args.use_rise_fall,
        motion_backbone=args.motion_backbone,
        use_au=args.use_au,
        au_schema=args.au_schema,
        use_gamdss=args.use_gamdss,
    )
    save_json(os.path.join(run_dir, "source_split.json"), split_metadata)
    print(json.dumps(split_metadata, indent=2))

    # The existing MA3D constructor loads its frozen landmark checkpoint.
    model = MA3D(**model_config).to(device)
    ce_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    ma_criterion = MarginAwareCELoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    start_epoch = 0
    best_val_uf1 = -1.0
    best_epoch = -1
    best_source_metrics: Dict[str, float] = {}
    if args.resume_checkpoint:
        checkpoint = _load_checkpoint(args.resume_checkpoint, device)
        _validate_resume(checkpoint, model_config)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_uf1 = float(checkpoint.get("best_val_uf1", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        best_source_metrics = dict(checkpoint.get("best_source_metrics", {}))
        print(f"Resumed from epoch {start_epoch}; best source-val UF1={best_val_uf1:.4f}")

    wandb = None
    if args.use_wandb:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise ImportError("--use_wandb was set, but wandb is not installed") from exc
        wandb = wandb_module
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_run_name or run_name,
            mode=args.wandb_mode,
            config=command_config,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("source/*", step_metric="epoch")
        if args.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=100)

    try:
        for epoch in range(start_epoch, args.epochs):
            started = time.time()
            train_loss, train_acc, train_labels, train_preds = train_one_epoch(
                model,
                train_loader,
                ce_criterion,
                lsce_criterion,
                ma_criterion,
                optimizer,
                device,
                epoch,
                args.epochs,
                aux_loss_weight=args.aux_loss_weight,
            )
            val_loss, val_acc, val_labels, val_preds = validate(
                model, val_loader, ce_criterion, device, epoch, args.epochs
            )
            train_metrics = _source_metrics(train_labels, train_preds)
            val_metrics = _source_metrics(val_labels, val_preds)
            is_best = val_metrics["UF1"] > best_val_uf1

            if is_best:
                best_val_uf1 = val_metrics["UF1"]
                best_epoch = epoch + 1
                best_source_metrics = val_metrics

            # Advance the scheduler before serializing so resume starts from
            # exactly the same LR that an uninterrupted next epoch would use.
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            state = {
                "format_version": 1,
                "protocol": "source_train_val_cross_dataset",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_uf1": best_val_uf1,
                "best_source_metrics": best_source_metrics,
                "model_config": model_config,
                "input_config": input_config,
                "source": source,
                "source_root": source_root,
                "source_split": split_metadata,
                "seed": args.seed,
            }
            if is_best:
                torch.save(state, best_path)
                np.save(os.path.join(run_dir, "best_source_val_labels.npy"), val_labels)
                np.save(os.path.join(run_dir, "best_source_val_preds.npy"), val_preds)
            torch.save(state, last_path)
            elapsed_min = (time.time() - started) / 60.0
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} | lr={lr:.3e} | "
                f"train loss={train_loss:.4f} UF1={train_metrics['UF1']:.4f} | "
                f"source-val loss={val_loss:.4f} UF1={val_metrics['UF1']:.4f} "
                f"UAR={val_metrics['UAR']:.4f} WAR={val_metrics['WAR']:.4f} | "
                f"best={best_val_uf1:.4f}@{best_epoch} | {elapsed_min:.2f} min"
            )

            if wandb is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "source/lr": lr,
                        "source/train_loss": train_loss,
                        "source/train_WAR": train_acc,
                        "source/train_UF1": train_metrics["UF1"],
                        "source/train_UAR": train_metrics["UAR"],
                        "source/val_loss": val_loss,
                        "source/val_WAR": val_acc,
                        "source/val_UF1": val_metrics["UF1"],
                        "source/val_UAR": val_metrics["UAR"],
                        "source/best_val_UF1": best_val_uf1,
                        "source/epoch_time_min": elapsed_min,
                    }
                )
                if is_best:
                    wandb.run.summary["best_source_val_UF1"] = best_val_uf1
                    wandb.run.summary["best_source_val_UAR"] = val_metrics["UAR"]
                    wandb.run.summary["best_source_val_WAR"] = val_metrics["WAR"]
                    wandb.run.summary["best_epoch"] = best_epoch

            if args.early_stop_on_perfect_val and val_acc >= 1.0:
                print("Early stop: source validation WAR reached 1.0")
                break

        if not os.path.exists(best_path):
            raise RuntimeError("Training completed without writing a best checkpoint")

        # Target data is intentionally loaded only after model selection is over.
        best_checkpoint = _load_checkpoint(best_path, device)
        model.load_state_dict(best_checkpoint["model"])
        best_epoch = int(best_checkpoint["best_epoch"])
        print(f"\nLoaded source-selected checkpoint: epoch={best_epoch}, path={best_path}")

        result_rows = []
        for target_key, target_root in targets:
            print(f"\n=== Unseen target: {DATASET_SPECS[target_key].display_name} ===")
            target_loader, target_metadata = build_target_loader(
                target_key,
                target_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
                n_roi=args.n_roi,
                use_rise_fall=args.use_rise_fall,
                motion_backbone=args.motion_backbone,
                use_au=args.use_au,
                au_schema=args.au_schema,
            )
            prediction = predict_loader(model, target_loader, ce_criterion, device)
            metrics = compute_metrics(prediction["labels"], prediction["preds"])
            target_dir = os.path.join(run_dir, f"target_{target_key}")
            os.makedirs(target_dir, exist_ok=True)
            save_prediction_csv(os.path.join(target_dir, "predictions.csv"), prediction)
            save_json(
                os.path.join(target_dir, "metrics.json"),
                {
                    "source": source,
                    "target": target_key,
                    "best_epoch": best_epoch,
                    "seed": args.seed,
                    "target_metadata": target_metadata,
                    "metrics": metrics,
                },
            )

            row = {
                "Source": DATASET_SPECS[source].display_name,
                "Target": DATASET_SPECS[target_key].display_name,
                "UF1": metrics["UF1"],
                "UAR": metrics["UAR"],
                "WAR": metrics["WAR"],
                "Samples": metrics["Samples"],
                "F1-Pos": metrics["F1-Pos"],
                "F1-Neg": metrics["F1-Neg"],
                "F1-Surp": metrics["F1-Surp"],
                "BestEpoch": best_epoch,
                "Seed": args.seed,
            }
            result_rows.append(row)
            print(
                f"UF1={row['UF1']:.4f} UAR={row['UAR']:.4f} WAR={row['WAR']:.4f} "
                f"N={row['Samples']} F1-Pos={row['F1-Pos']:.4f} "
                f"F1-Neg={row['F1-Neg']:.4f} F1-Surp={row['F1-Surp']:.4f}"
            )

            if wandb is not None:
                prefix = f"transfer/{target_key}"
                scalar_log = {
                    f"{prefix}/UF1": metrics["UF1"],
                    f"{prefix}/UAR": metrics["UAR"],
                    f"{prefix}/WAR": metrics["WAR"],
                    f"{prefix}/Samples": metrics["Samples"],
                    f"{prefix}/F1_Pos": metrics["F1-Pos"],
                    f"{prefix}/F1_Neg": metrics["F1-Neg"],
                    f"{prefix}/F1_Surp": metrics["F1-Surp"],
                }
                cm_fig = _plot_confusion(prediction["labels"], prediction["preds"])
                scalar_log[f"{prefix}/confusion_matrix"] = wandb.Image(cm_fig)
                wandb.log(scalar_log)
                plt.close(cm_fig)
                for key, value in scalar_log.items():
                    if not key.endswith("confusion_matrix"):
                        wandb.run.summary[key] = value

                table_columns = [
                    "sample_id",
                    "label",
                    "prediction",
                    "logit_negative",
                    "logit_positive",
                    "logit_surprise",
                ]
                table_data = []
                for sample_id, label, pred, logits in zip(
                    prediction["sample_ids"],
                    prediction["labels"],
                    prediction["preds"],
                    prediction["logits"],
                ):
                    table_data.append(
                        [sample_id, int(label), int(pred), *[float(x) for x in logits]]
                    )
                wandb.log({f"{prefix}/predictions": wandb.Table(columns=table_columns, data=table_data)})

        result_path = os.path.join(run_dir, "transfer_results.csv")
        write_result_table(result_path, result_rows)
        print(f"\nSaved result table: {result_path}")

        if wandb is not None:
            artifact = wandb.Artifact(
                name=_safe_name(f"ma3d_{source}_{run_name}"),
                type="model",
                metadata={
                    "source": source,
                    "best_epoch": best_epoch,
                    "seed": args.seed,
                    "input_config": input_config,
                },
            )
            artifact.add_file(best_path)
            wandb.log_artifact(artifact)
    finally:
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
