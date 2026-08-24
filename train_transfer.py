"""Train MA3D on the full source dataset and evaluate targets every epoch.

The fixed final checkpoint (normally epoch 100) is the primary cross-dataset
result. For exploratory analysis, the script also records an explicitly
labelled best-target/oracle checkpoint for each target, selected by target UF1.
Target labels never contribute gradients, but the oracle result is not an
unseen-test estimate because its epoch is selected on the target itself.
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
    build_full_source_loader,
    build_target_loader,
    compute_metrics,
    normalize_dataset_key,
    predict_loader,
    save_json,
    save_prediction_csv,
    validate_au_protocol,
    write_result_table,
)
from engine import train_one_epoch
from loss_function.loss import LabelSmoothingCrossEntropy, MarginAwareCELoss
from models.MA3D import MA3D


PROTOCOL_NAME = "full_source_cross_dataset_with_target_oracle"
TARGET_SELECTION_METRIC = "UF1"


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
        "MA3D full-source training -> cross-dataset final + best-target/oracle evaluation"
    )

    parser.add_argument("--source", required=True, help="4dme or casme2")
    parser.add_argument("--source_root", required=True)
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="DATASET=ROOT",
        help="Repeat for each target, e.g. --target casme2=/path/to/data",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_gamma", type=float, default=0.98)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_sampler", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--model_type", choices=["small", "base", "large"], default="small")
    parser.add_argument("--n_roi", type=int, default=6)
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

    parser.add_argument("--use_au", action="store_true")
    parser.add_argument("--au_embed_dim", type=int, default=128)

    parser.add_argument("--output_dir", default="transfer_runs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Path to a last_checkpoint.pth produced by this script",
    )

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
    except TypeError:
        return torch.load(path, map_location=device)


def _scalar_metrics(labels, preds) -> Dict[str, float]:
    metrics = compute_metrics(labels, preds)
    return {
        "UF1": float(metrics["UF1"]),
        "UAR": float(metrics["UAR"]),
        "WAR": float(metrics["WAR"]),
    }


def _target_summary(metrics: Mapping[str, object], epoch: int) -> Dict[str, object]:
    return {
        "epoch": int(epoch),
        "UF1": float(metrics["UF1"]),
        "UAR": float(metrics["UAR"]),
        "WAR": float(metrics["WAR"]),
        "Samples": int(metrics["Samples"]),
        "F1-Pos": float(metrics["F1-Pos"]),
        "F1-Neg": float(metrics["F1-Neg"]),
        "F1-Surp": float(metrics["F1-Surp"]),
    }


def _format_metrics(metrics: Mapping[str, object]) -> str:
    return (
        f"UF1={float(metrics['UF1']):.4f} UAR={float(metrics['UAR']):.4f} "
        f"WAR={float(metrics['WAR']):.4f} N={int(metrics['Samples'])} "
        f"F1-Pos={float(metrics['F1-Pos']):.4f} "
        f"F1-Neg={float(metrics['F1-Neg']):.4f} "
        f"F1-Surp={float(metrics['F1-Surp']):.4f}"
    )


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
    protocol = checkpoint.get("protocol")
    if protocol != PROTOCOL_NAME:
        raise ValueError(
            f"Resume checkpoint protocol is {protocol!r}, expected {PROTOCOL_NAME!r}. "
            "A checkpoint from the former source-validation protocol cannot be resumed here."
        )
    saved = checkpoint.get("model_config")
    if saved != current:
        raise ValueError(
            "Resume checkpoint model_config does not match the current command.\n"
            f"saved={saved}\ncurrent={current}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = get_args(argv)
    source = normalize_dataset_key(args.source)
    if source == "smic_hs":
        raise ValueError("SMIC-HS is target-only in this protocol; choose 4dme or casme2 as source")
    targets = parse_targets(args.target)
    target_keys = [key for key, _ in targets]
    if source in target_keys:
        raise ValueError("Source dataset must not also be listed as a target")

    au_schema = "common15"
    au_dim = 0
    if args.use_au:
        au_dim = validate_au_protocol(source, target_keys, au_schema)

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
    last_path = os.path.join(run_dir, "last_checkpoint.pth")

    if os.path.isdir(run_dir) and os.listdir(run_dir) and not args.resume_checkpoint:
        raise FileExistsError(
            f"Run directory {run_dir!r} is not empty. Use a new --run_name or "
            "pass --resume_checkpoint; existing results are not overwritten implicitly."
        )
    os.makedirs(run_dir, exist_ok=True)

    model_config = _build_model_config(args, au_dim=au_dim)
    input_config = {
        "num_classes": 3,
        "class_names": CLASS_NAMES,
        "n_roi": args.n_roi,
        "use_rise_fall": args.use_rise_fall,
        "rise_fall_mode": args.rise_fall_mode,
        "motion_backbone": args.motion_backbone,
        "use_gamdss_source_train_only": args.use_gamdss,
        "use_au": args.use_au,
        "au_schema": au_schema if args.use_au else None,
        "au_dim": au_dim if args.use_au else None,
        "common_au_vocab": COMMON15_AU_VOCAB if args.use_au else None,
    }
    command_config = {
        **vars(args),
        "protocol": PROTOCOL_NAME,
        "source_training": "full_dataset",
        "target_evaluated_each_epoch": True,
        "primary_result": "last_checkpoint",
        "oracle_selection_metric": TARGET_SELECTION_METRIC,
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
    print(f"Protocol: full source for {args.epochs} epochs")
    print(f"Primary result: last checkpoint (epoch {args.epochs})")
    print(f"Exploratory result: best-target/oracle selected by target {TARGET_SELECTION_METRIC}")
    print(f"Source: {DATASET_SPECS[source].display_name} -> {source_root}")
    print(
        "Targets: "
        + ", ".join(f"{DATASET_SPECS[key].display_name} -> {root}" for key, root in targets)
    )
    if args.use_au:
        print(f"AU: schema={au_schema}, au_dim={au_dim}")
    else:
        print("AU: disabled")

    train_loader, source_metadata = build_full_source_loader(
        source,
        source_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_sampler=args.use_sampler,
        n_roi=args.n_roi,
        use_rise_fall=args.use_rise_fall,
        motion_backbone=args.motion_backbone,
        use_au=args.use_au,
        au_schema=au_schema,
        use_gamdss=args.use_gamdss,
    )
    save_json(os.path.join(run_dir, "source_dataset.json"), source_metadata)
    print(json.dumps(source_metadata, indent=2))

    target_loaders = {}
    target_metadata = {}
    for target_key, target_root in targets:
        loader, metadata = build_target_loader(
            target_key,
            target_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            n_roi=args.n_roi,
            use_rise_fall=args.use_rise_fall,
            motion_backbone=args.motion_backbone,
            use_au=args.use_au,
            au_schema=au_schema,
        )
        target_loaders[target_key] = loader
        target_metadata[target_key] = metadata
        target_dir = os.path.join(run_dir, f"target_{target_key}")
        os.makedirs(target_dir, exist_ok=True)
        save_json(os.path.join(target_dir, "dataset.json"), metadata)

    model = MA3D(**model_config).to(device)
    ce_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    ma_criterion = MarginAwareCELoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    start_epoch = 0
    best_targets: Dict[str, Dict[str, object]] = {}
    if args.resume_checkpoint:
        checkpoint = _load_checkpoint(args.resume_checkpoint, device)
        _validate_resume(checkpoint, model_config)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_targets = {
            str(key): dict(value)
            for key, value in dict(checkpoint.get("best_targets", {})).items()
        }
        print(f"Resumed from completed epoch {start_epoch}")

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
        wandb.define_metric("target_oracle/*", step_metric="epoch")
        if args.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=100)

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_number = epoch + 1
            started = time.time()
            lr_used = optimizer.param_groups[0]["lr"]
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
            train_metrics = _scalar_metrics(train_labels, train_preds)
            scheduler.step()

            improved_targets = []
            target_epoch_metrics = {}
            target_log_parts = []
            for target_key, _ in targets:
                prediction = predict_loader(
                    model, target_loaders[target_key], ce_criterion, device
                )
                metrics = compute_metrics(prediction["labels"], prediction["preds"])
                target_epoch_metrics[target_key] = metrics
                previous = best_targets.get(target_key)
                if previous is None or float(metrics[TARGET_SELECTION_METRIC]) > float(
                    previous[TARGET_SELECTION_METRIC]
                ):
                    best_targets[target_key] = _target_summary(metrics, epoch_number)
                    improved_targets.append(target_key)
                    target_dir = os.path.join(run_dir, f"target_{target_key}")
                    save_prediction_csv(
                        os.path.join(target_dir, "best_target_predictions.csv"), prediction
                    )
                    save_json(
                        os.path.join(target_dir, "best_target_selection.json"),
                        {
                            "label": "best-target/oracle",
                            "selected_by": f"target_{TARGET_SELECTION_METRIC}",
                            "epoch": epoch_number,
                            "metrics": metrics,
                        },
                    )
                best = best_targets[target_key]
                target_log_parts.append(
                    f"{DATASET_SPECS[target_key].display_name} target "
                    f"UF1={metrics['UF1']:.4f} best={best['UF1']:.4f}@{best['epoch']}"
                )

            state = {
                "format_version": 2,
                "protocol": PROTOCOL_NAME,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_targets": best_targets,
                "target_selection_metric": TARGET_SELECTION_METRIC,
                "model_config": model_config,
                "input_config": input_config,
                "source": source,
                "source_root": source_root,
                "source_dataset": source_metadata,
                "seed": args.seed,
            }
            torch.save(state, last_path)
            for target_key in improved_targets:
                best_target_path = os.path.join(
                    run_dir, f"best_target_{target_key}_checkpoint.pth"
                )
                target_state = dict(state)
                target_state["selected_target"] = target_key
                target_state["selected_target_metrics"] = best_targets[target_key]
                torch.save(target_state, best_target_path)

            elapsed_min = (time.time() - started) / 60.0
            print(
                f"Epoch {epoch_number:03d}/{args.epochs} | lr={lr_used:.3e} | "
                f"full-source loss={train_loss:.4f} UF1={train_metrics['UF1']:.4f} "
                f"UAR={train_metrics['UAR']:.4f} WAR={train_metrics['WAR']:.4f} | "
                + " | ".join(target_log_parts)
                + f" | {elapsed_min:.2f} min"
            )

            if wandb is not None:
                wandb_log = {
                    "epoch": epoch_number,
                    "source/lr": lr_used,
                    "source/train_loss": train_loss,
                    "source/train_WAR": train_acc,
                    "source/train_UF1": train_metrics["UF1"],
                    "source/train_UAR": train_metrics["UAR"],
                    "source/epoch_time_min": elapsed_min,
                }
                for target_key, metrics in target_epoch_metrics.items():
                    prefix = f"target_oracle/{target_key}"
                    wandb_log.update(
                        {
                            f"{prefix}/UF1": metrics["UF1"],
                            f"{prefix}/UAR": metrics["UAR"],
                            f"{prefix}/WAR": metrics["WAR"],
                            f"{prefix}/F1_Pos": metrics["F1-Pos"],
                            f"{prefix}/F1_Neg": metrics["F1-Neg"],
                            f"{prefix}/F1_Surp": metrics["F1-Surp"],
                            f"{prefix}/best_UF1": best_targets[target_key]["UF1"],
                        }
                    )
                wandb.log(wandb_log)
                for target_key in improved_targets:
                    wandb.run.summary[f"best_target_oracle/{target_key}/epoch"] = best_targets[
                        target_key
                    ]["epoch"]
                    for metric_name in ("UF1", "UAR", "WAR", "F1-Pos", "F1-Neg", "F1-Surp"):
                        wandb.run.summary[
                            f"best_target_oracle/{target_key}/{metric_name}"
                        ] = best_targets[target_key][metric_name]

        if not os.path.exists(last_path):
            raise RuntimeError("Training completed without writing last_checkpoint.pth")

        last_checkpoint = _load_checkpoint(last_path, device)
        final_epoch = int(last_checkpoint["epoch"]) + 1
        best_targets = {
            str(key): dict(value)
            for key, value in dict(last_checkpoint.get("best_targets", {})).items()
        }
        result_rows = []

        print("\n============================================================")
        print("FINAL CROSS-DATASET RESULTS")
        print("============================================================")
        for target_key, _ in targets:
            target_name = DATASET_SPECS[target_key].display_name
            target_dir = os.path.join(run_dir, f"target_{target_key}")

            model.load_state_dict(last_checkpoint["model"])
            final_prediction = predict_loader(
                model, target_loaders[target_key], ce_criterion, device
            )
            final_metrics = compute_metrics(
                final_prediction["labels"], final_prediction["preds"]
            )
            save_prediction_csv(
                os.path.join(target_dir, "last_checkpoint_predictions.csv"),
                final_prediction,
            )

            best_target_path = os.path.join(
                run_dir, f"best_target_{target_key}_checkpoint.pth"
            )
            if target_key not in best_targets or not os.path.exists(best_target_path):
                raise RuntimeError(
                    f"Missing best-target/oracle checkpoint or summary for {target_key}"
                )
            best_checkpoint = _load_checkpoint(best_target_path, device)
            best_epoch = int(best_targets[target_key]["epoch"])
            model.load_state_dict(best_checkpoint["model"])
            best_prediction = predict_loader(
                model, target_loaders[target_key], ce_criterion, device
            )
            best_metrics = compute_metrics(
                best_prediction["labels"], best_prediction["preds"]
            )
            save_prediction_csv(
                os.path.join(target_dir, "best_target_predictions.csv"),
                best_prediction,
            )
            save_json(
                os.path.join(target_dir, "metrics.json"),
                {
                    "source": source,
                    "target": target_key,
                    "seed": args.seed,
                    "target_metadata": target_metadata[target_key],
                    "primary_result": {
                        "label": "last-checkpoint",
                        "epoch": final_epoch,
                        "checkpoint": last_path,
                        "metrics": final_metrics,
                    },
                    "exploratory_result": {
                        "label": "best-target/oracle",
                        "selected_by": f"target_{TARGET_SELECTION_METRIC}",
                        "epoch": best_epoch,
                        "checkpoint": best_target_path,
                        "metrics": best_metrics,
                    },
                },
            )

            print(f"\nTarget: {target_name}")
            print(
                f"  LAST CHECKPOINT (primary, epoch {final_epoch}): "
                + _format_metrics(final_metrics)
            )
            print(
                f"  BEST-TARGET/ORACLE (target UF1, epoch {best_epoch}): "
                + _format_metrics(best_metrics)
            )

            common_row = {
                "Source": DATASET_SPECS[source].display_name,
                "Target": target_name,
                "Samples": final_metrics["Samples"],
                "Seed": args.seed,
            }
            result_rows.append(
                {
                    **common_row,
                    "Checkpoint": "last_checkpoint",
                    "Epoch": final_epoch,
                    "SelectionMetric": "fixed_final_epoch",
                    **{
                        key: final_metrics[key]
                        for key in ("UF1", "UAR", "WAR", "F1-Pos", "F1-Neg", "F1-Surp")
                    },
                }
            )
            result_rows.append(
                {
                    **common_row,
                    "Checkpoint": "best_target_oracle",
                    "Epoch": best_epoch,
                    "SelectionMetric": f"target_{TARGET_SELECTION_METRIC}",
                    **{
                        key: best_metrics[key]
                        for key in ("UF1", "UAR", "WAR", "F1-Pos", "F1-Neg", "F1-Surp")
                    },
                }
            )

            if wandb is not None:
                final_prefix = f"final/{target_key}/last_checkpoint"
                best_prefix = f"final/{target_key}/best_target_oracle"
                for metric_name in ("UF1", "UAR", "WAR", "F1-Pos", "F1-Neg", "F1-Surp"):
                    wandb.run.summary[f"{final_prefix}/{metric_name}"] = final_metrics[
                        metric_name
                    ]
                    wandb.run.summary[f"{best_prefix}/{metric_name}"] = best_metrics[
                        metric_name
                    ]
                wandb.run.summary[f"{final_prefix}/epoch"] = final_epoch
                wandb.run.summary[f"{best_prefix}/epoch"] = best_epoch

                final_cm = _plot_confusion(
                    final_prediction["labels"], final_prediction["preds"]
                )
                best_cm = _plot_confusion(
                    best_prediction["labels"], best_prediction["preds"]
                )
                wandb.log(
                    {
                        f"{final_prefix}/confusion_matrix": wandb.Image(final_cm),
                        f"{best_prefix}/confusion_matrix": wandb.Image(best_cm),
                    }
                )
                plt.close(final_cm)
                plt.close(best_cm)

        result_path = os.path.join(run_dir, "transfer_results.csv")
        write_result_table(result_path, result_rows)
        print(f"\nSaved result table: {result_path}")

        if wandb is not None:
            artifact = wandb.Artifact(
                name=_safe_name(f"ma3d_{source}_{run_name}"),
                type="model",
                metadata={
                    "protocol": PROTOCOL_NAME,
                    "source": source,
                    "final_epoch": final_epoch,
                    "target_selection_metric": TARGET_SELECTION_METRIC,
                    "seed": args.seed,
                    "input_config": input_config,
                },
            )
            artifact.add_file(last_path)
            for target_key, _ in targets:
                artifact.add_file(
                    os.path.join(run_dir, f"best_target_{target_key}_checkpoint.pth")
                )
            wandb.log_artifact(artifact)
    finally:
        if wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
