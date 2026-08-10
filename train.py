import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
import torch
import torch.optim as optim
from models.MA3D import MA3D
from loss_function.loss import MarginAwareCELoss, LabelSmoothingCrossEntropy
from models.sam import SAM

from build_dataloader import get_dataloaders, get_loso_dataloaders
from engine import train_one_epoch, validate
import time
import shutil
import argparse

import wandb
from collections import Counter
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_args():
    parser = argparse.ArgumentParser("SFER Training")

    parser.add_argument("--seed", type=int, default=42)

    # Dataset
    parser.add_argument("--data_type", default="RAF-DB",
                        choices=["RAF-DB", "VKIST", "Cheo", "FerPlus", "Caers",
                                 "CheoFaMo", "4DME_MOTION", "CASME2_MOTION",
                                 "SMIC_HS_MOTION"])
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--class_names", type=str, default=None,
                        help="Comma-separated class names in label-index order, "
                             "e.g. 'Negative,Positive,Surprise,Repression,Others'. "
                             "Defaults to '0','1',... if not given.")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_type", type=str, default="large",
                    choices=["small", "base", "large"])

    # MA3D architecture (motion / landmark-modulator / appearance-context)
    parser.add_argument("--n_roi", type=int, default=3,
                        help="Number of ROI in the flow map (eyebrow, eye, mouth). "
                             "Must match the flow_map.npy produced by "
                             "run_inference_flow.py.")
    parser.add_argument("--landmark_embed_dim", type=int, default=256,
                        help="Embedding dim of the landmark-derived FiLM "
                             "condition vector (LandmarkPriorEncoder output).")
    parser.add_argument("--num_film_blocks", type=int, default=5,
                        help="Number of FiLM (LandmarkModulationFusion) blocks "
                             "applied to the motion feature map.")
    parser.add_argument("--use_rise_fall", action="store_true")
    parser.add_argument("--rise_fall_mode", type=str, default="feature_gate",
                        choices=["feature_gate", "decision_level"],
                        help="Only has an effect when --use_rise_fall is set. "
                             "'feature_gate': this codebase's own design (NOT a "
                             "faithful port of any paper) -- rise/fall features "
                             "combined via a cosine-similarity agreement map "
                             "BEFORE landmark FiLM + pyramid_fuse; pyramid_fuse "
                             "runs ONCE (cheaper). "
                             "'decision_level': matches GAMDSS's actual BDRT "
                             "class AND its real training script (verified) -- "
                             "NO feature-level fusion; landmark FiLM + "
                             "pyramid_fuse + head run TWICE (once per phase, "
                             "shared weights), but the FINAL PREDICTION is the "
                             "rise-phase output ONLY -- fall-phase output is "
                             "used purely as an auxiliary loss term, never "
                             "averaged in (matches GAMDSS exactly: predictions "
                             "come from `ALL`, never from `s`). Roughly 2x the "
                             "compute of feature_gate mode. See MA3D.py class "
                             "docstring for full detail on both.")
    parser.add_argument("--aux_loss_weight", type=float, default=0.2)
    parser.add_argument("--motion_backbone", type=str, default="cnn",
                        choices=["cnn", "rmt"])
    parser.add_argument("--use_au", action="store_true")
    parser.add_argument("--au_dim", type=int, default=36)
    parser.add_argument("--au_embed_dim", type=int, default=128)
    parser.add_argument("--use_gamdss", action="store_true",
                        help="4DME_MOTION/CASME2_MOTION only: TRAIN view reads the GAMDSS "
                             "dynamic-frame-reselection-corrected files "
                             "(inputs_gamdss.png, onset_gamdss.png, "
                             "flow_map_gamdss.npy, flow_map_fall_gamdss.npy). "
                             "VAL view is UNAFFECTED -- always reads the "
                             "original files, by design (only train gets "
                             "relabeled, evaluation stays on official "
                             "annotations -- see chat notes).")

    parser.add_argument("--use_sampler", action="store_true",
                        help="Use a class-balanced WeightedRandomSampler for "
                             "training instead of plain shuffling.")
    parser.add_argument("--loso_debug_subject", type=str, default=None,
                        help="LOSO ME datasets only (4DME_MOTION/CASME2_MOTION/"
                             "SMIC_HS_MOTION): restrict the LOSO loop to a "
                             "single held-out subject, for quick smoke-testing "
                             "of the pipeline without running all folds.")
    parser.add_argument("--early_stop_on_perfect_val", action="store_true",
                        help="Stop training the current fold as soon as "
                             "val_acc reaches 100%% right after saving the "
                             "best checkpoint/predictions.")

    # Logging
    parser.add_argument("--log_file", type=str, default="log.txt")

    # Checkpoint / Resume
    parser.add_argument("--resume_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_name", type=str, default="last.pth")
    parser.add_argument("--resume", action="store_true")

    # Periodic backup
    parser.add_argument("--backup_dir", type=str, default="/kaggle/working",
                        help="Directory to periodically backup checkpoint")
    parser.add_argument("--backup_every", type=int, default=5,
                        help="Backup checkpoint every N epochs")

    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MA3D-micro")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_watch_model", action="store_true")

    return parser.parse_args()


def compute_extra_metrics(y_true, y_pred, num_classes):
    """UF1 = macro-F1, UAR = macro-recall, weighted-F1, classification_report"""
    labels = list(range(num_classes))
    uf1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return uf1, uar, weighted_f1, report


def plot_confusion_matrix(y_true, y_pred, num_classes, class_names=None):
    labels = list(range(num_classes))
    names = class_names if class_names else [str(i) for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def save_checkpoint(state, resume_path, backup_dir, tag, is_periodic=False, epoch=None):
    """
    tag: fold identifier (e.g. held-out subject id) used to prefix backup
    filenames, so different LOSO folds writing to the SAME backup_dir don't
    overwrite each other. tag="" (empty string) for the non-LOSO case.
    """
    torch.save(state, resume_path)

    prefix = f"{tag}_" if tag else ""

    if is_periodic:
        for f in os.listdir(backup_dir):
            if f.startswith(f"{prefix}4dme_epoch") and f.endswith(".pth"):
                os.remove(os.path.join(backup_dir, f))
        backup_name = f"{prefix}4dme_epoch{epoch}.pth"
    else:
        backup_name = f"{prefix}4dme_best.pth"

    shutil.copy(resume_path, os.path.join(backup_dir, backup_name))


def run_fold(args, train_loader, val_loader, device, fold_tag=None):
    """
    Runs one full training run (all epochs) on the given train/val loaders,
    then returns (best_val_acc, best_val_uf1, uar_at_best_uf1).

    fold_tag: None for a regular (non-LOSO) run; a string (e.g. the
    held-out subject id) when called from the LOSO loop in main() -- used
    to keep checkpoints/logs/wandb runs from different folds separate.
    """
    tag_str = str(fold_tag) if fold_tag is not None else ""
    resume_name = f"{tag_str}_{args.resume_name}" if fold_tag is not None else args.resume_name
    resume_path = os.path.join(args.resume_dir, resume_name)

    # ---- Weights & Biases setup ----
    use_wandb = args.use_wandb
    wandb_run_id = args.wandb_run_id

    if use_wandb:
        base_name = args.wandb_run_name or f"{args.data_type}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_name = f"{base_name}_{tag_str}" if fold_tag is not None else base_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=wandb_run_id,
            resume="allow",
            mode=args.wandb_mode,
            config=vars(args),
            reinit=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    log_f = None
    if args.log_file is not None:
        os.makedirs("log", exist_ok=True)
        log_name = f"{tag_str}_{args.log_file}" if fold_tag is not None else args.log_file
        log_path = os.path.join("log", log_name)
        log_f = open(log_path, "a")
        log_f.write(f"resume path / checkpoint path: {resume_path}\n")
        log_f.write(f"Batch_size: {args.batch_size}\n")
        if fold_tag is not None:
            log_f.write(f"LOSO fold -- held-out subject: {fold_tag}\n")
        log_f.flush()
        log_f.write(
            f"{'Epoch':^6} {'LR':^12} {'Train_Loss':^12} {'Train_Acc':^10} "
            f"{'Val_Loss':^12} {'Val_Acc':^10} {'Time(min)':^10}\n"
        )
        log_f.flush()

    class_names = args.class_names.split(",") if args.class_names else None
    if class_names and len(class_names) != args.num_classes:
        print(f"[WARN] --class_names has {len(class_names)} names but num_classes={args.num_classes}, "
              f"-> SKIP")
        class_names = None

    model = MA3D(num_classes=args.num_classes, type=args.model_type,
                  n_roi=args.n_roi, landmark_embed_dim=args.landmark_embed_dim,
                  num_film_blocks=args.num_film_blocks,
                  use_rise_fall=args.use_rise_fall,
                  rise_fall_mode=args.rise_fall_mode,
                  motion_backbone=args.motion_backbone,
                  use_au=args.use_au,
                  au_dim=args.au_dim,
                  au_embed_dim=args.au_embed_dim).to(device)

    if use_wandb and args.wandb_watch_model:
        wandb.watch(model, log="all", log_freq=100)

    CE_criterion  = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    MA_criterion = MarginAwareCELoss().to(device)

    base_optimizer = optim.AdamW
    # optimizer = SAM(model.parameters(), base_optimizer,
    #                 lr=args.lr, weight_decay=args.weight_decay,
    #                 rho=0.5, adaptive=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    start_epoch = 0
    best_val_acc = 0.0
    best_val_uf1 = -1.0
    uar_at_best = 0.0

    if args.resume and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_uf1 = checkpoint.get("best_val_uf1", -1.0)
        uar_at_best = checkpoint.get("uar_at_best", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}, best_val_uf1={best_val_uf1:.4f}")
        if wandb_run_id:
            print(f"wandb run_id: {wandb_run_id}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_loss, train_acc, train_labels, train_preds = train_one_epoch(
            model, train_loader, CE_criterion, lsce_criterion,
            MA_criterion, optimizer, device, epoch, args.epochs,
            aux_loss_weight=args.aux_loss_weight,
        )

        val_loss, val_acc, val_labels, val_preds = validate(
            model, val_loader, CE_criterion, device, epoch, args.epochs,
        )

        val_uf1, val_uar, val_weighted_f1, val_report = compute_extra_metrics(
            val_labels, val_preds, args.num_classes
        )

        train_uf1, train_uar, train_weighted_f1, train_report = compute_extra_metrics(
            train_labels, train_preds, args.num_classes
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        epoch_time = (time.time() - epoch_start_time) / 60.0

        # Use UF1 (macro-F1) for the "best" metric -- acc is not a good
        # criterion for imbalanced datasets, especially when the majority
        # class dominates. best_val_acc is still tracked in parallel.
        is_best = val_uf1 > best_val_uf1

        state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": val_acc if is_best else best_val_acc,
            "best_val_uf1": best_val_uf1 if not is_best else val_uf1,
            "uar_at_best":  uar_at_best if not is_best else val_uar,
        }

        best_val_acc = max(best_val_acc, val_acc)

        # save best ckpt (UF1 / macro-F1)
        if is_best:
            best_val_uf1 = val_uf1
            uar_at_best = val_uar
            save_checkpoint(state, resume_path, args.backup_dir, tag_str, is_periodic=False)
            print(f"[Best] epoch={epoch+1}  val_acc={val_acc*100:.2f}%  "
                  f"UF1={val_uf1:.4f}  UAR={val_uar:.4f}")

            np.save(os.path.join(args.backup_dir, f"{tag_str}_best_val_labels.npy" if tag_str else "best_val_labels.npy"), val_labels)
            np.save(os.path.join(args.backup_dir, f"{tag_str}_best_val_preds.npy" if tag_str else "best_val_preds.npy"), val_preds)

            if use_wandb:
                wandb.run.summary["best_val_acc"]  = best_val_acc
                wandb.run.summary["best_val_uf1"]  = best_val_uf1
                wandb.run.summary["best_val_uar"]  = val_uar
                wandb.run.summary["best_epoch"]    = epoch + 1

                cm_fig = plot_confusion_matrix(val_labels, val_preds, args.num_classes, class_names)
                wandb.log({"val/confusion_matrix_best": wandb.Image(cm_fig), "epoch": epoch + 1})
                plt.close(cm_fig)

            if log_f:
                log_f.write(f"BEST\tval_acc={val_acc*100:.2f}\tUF1={val_uf1:.4f}\tUAR={val_uar:.4f}\tF1={val_weighted_f1:.4f}\n")
                per_class = {
                    (class_names[int(k)] if class_names and k.isdigit() else k): v
                    for k, v in val_report.items()
                    if k not in ("accuracy", "macro avg", "weighted avg")
                }
                log_f.write(f"\tPer-class: {per_class}\n")
                log_f.flush()

            # Early stop THIS fold once val_acc hits 100%: the held-out
            # val set is fixed, so every sample is already predicted
            # correctly
            if args.early_stop_on_perfect_val and val_acc >= 1.0:
                print(f"[EarlyStop] epoch={epoch+1}  fold={fold_tag}  "
                      f"val_acc=100% on {len(val_labels)} held-out sample(s) "
                      f"-- stopping this fold early (pooled result can't "
                      f"improve further).")
                if log_f:
                    log_f.write(f"EARLYSTOP\tepoch={epoch+1}\tval_acc=100%\n")
                    log_f.flush()
                break

        # backup each N epoch
        if (epoch + 1) % args.backup_every == 0:
            save_checkpoint(state, resume_path, args.backup_dir, tag_str,
                            is_periodic=True, epoch=epoch+1)
            print(f"[Backup] epoch={epoch+1} -> {args.backup_dir}/{tag_str}_4dme_epoch{epoch+1}.pth")

        # wandb log
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "lr": lr,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/uf1": train_uf1,
                "train/uar": train_uar,
                "train/weighted_f1": train_weighted_f1,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/uf1": val_uf1,
                "val/uar": val_uar,
                "val/weighted_f1": val_weighted_f1,
                "epoch_time_min": epoch_time,
                "best_val_acc": best_val_acc,
                "best_val_uf1": best_val_uf1,
            })

        if log_f:
            log_f.write(
                f"{epoch + 1:^6d} {lr:^12.8f} {train_loss:^12.4f} {train_acc * 100:^10.2f} "
                f"trainUF1={train_uf1:.4f} trainUAR={train_uar:.4f} "
                f"{val_loss:^12.4f} {val_acc * 100:^10.2f} "
                f"UF1={val_uf1:.4f} UAR={val_uar:.4f} {epoch_time:^10.2f}\n"
            )
            log_f.flush()

    print(f"\n[{fold_tag if fold_tag is not None else 'run'}] "
          f"Best validation accuracy: {best_val_acc * 100:.2f}%  |  "
          f"Best UF1 (macro-F1): {best_val_uf1:.4f}  |  UAR at best: {uar_at_best:.4f}")
    if log_f:
        log_f.write(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%  |  "
                    f"Best UF1: {best_val_uf1:.4f}  |  UAR at best: {uar_at_best:.4f}")
        log_f.close()

    if use_wandb:
        wandb.finish()

    if os.path.exists(resume_path):
        os.remove(resume_path)
    for f in os.listdir(args.backup_dir):
        if f.startswith(f"{tag_str}_4dme_epoch") and f.endswith(".pth"):
            os.remove(os.path.join(args.backup_dir, f))
        if f.startswith(f"{tag_str}_4dme_best") and f.endswith(".pth"):
            os.remove(os.path.join(args.backup_dir, f))

    return best_val_acc, best_val_uf1, uar_at_best


def main():
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.resume_dir, exist_ok=True)
    os.makedirs(args.backup_dir, exist_ok=True)

    if args.data_type in ("4DME_MOTION", "CASME2_MOTION", "SMIC_HS_MOTION"):
        _expected_num_classes = {
            "4DME_MOTION": 5, "CASME2_MOTION": 5, "SMIC_HS_MOTION": 3,
        }[args.data_type]
        if args.num_classes != _expected_num_classes:
            print(f"[WARN] data_type={args.data_type} but num_classes={args.num_classes} "
                  f"(expected {_expected_num_classes})")

        if args.use_au:
            if args.data_type == "SMIC_HS_MOTION":
                print(f"[WARN] SMIC_HS_MOTION has no AU annotations (no au.npy on disk) "
                      f"-- --use_au will make every sample fail FourDME_Dataset's "
                      f"required-file check, so the dataset will end up empty.")
            else:
                expected_au_dim = 36 if args.data_type == "4DME_MOTION" else 19
                if args.au_dim != expected_au_dim:
                    print(f"[WARN] data_type={args.data_type} with --use_au but "
                          f"au_dim={args.au_dim} (expected {expected_au_dim}) -- "
                          f"the au.npy files on disk won't match this shape.")

        splits = get_loso_dataloaders(args)
        print(f"[LOSO] {len(splits)} subject folds queued")

        fold_accs, fold_uf1s, fold_uars = [], [], []
        for fold_idx, (train_loader, val_loader, held_out) in enumerate(splits):
            print(f"\n=== Fold {fold_idx + 1}/{len(splits)} -- held-out subject: {held_out} ===")
            best_acc, best_uf1, best_uar = run_fold(
                args, train_loader, val_loader, device, fold_tag=held_out
            )
            fold_accs.append(best_acc)
            fold_uf1s.append(best_uf1)
            fold_uars.append(best_uar)

        fold_accs = np.array(fold_accs)
        fold_uf1s = np.array(fold_uf1s)
        fold_uars = np.array(fold_uars)

        summary = (
            f"\n=== LOSO summary across {len(splits)} folds ===\n"
            f"Acc : {fold_accs.mean() * 100:.2f}% (+/- {fold_accs.std() * 100:.2f})\n"
            f"UF1 : {fold_uf1s.mean():.4f} (+/- {fold_uf1s.std():.4f})\n"
            f"UAR : {fold_uars.mean():.4f} (+/- {fold_uars.std():.4f})\n"
        )
        print(summary)

        os.makedirs("log", exist_ok=True)
        with open(os.path.join("log", f"loso_summary_{args.log_file}"), "w") as f:
            f.write(f"Per-fold Acc: {fold_accs.tolist()}\n")
            f.write(f"Per-fold UF1: {fold_uf1s.tolist()}\n")
            f.write(f"Per-fold UAR: {fold_uars.tolist()}\n")
            f.write(summary)

    else:
        train_loader, val_loader = get_dataloaders(args)
        run_fold(args, train_loader, val_loader, device, fold_tag=None)


if __name__ == "__main__":
    main()