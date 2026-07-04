import os
import torch
import torch.optim as optim
from models.MA3D import MA3D
from loss_function.loss import MarginAwareCELoss, LabelSmoothingCrossEntropy
from models.sam import SAM

from build_dataloader import get_dataloaders
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
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser("SFER Training")

    parser.add_argument("--seed", type=int, default=42)
    # Dataset
    parser.add_argument("--data_type", default="RAF-DB", choices=["RAF-DB", "VKIST", "Cheo", "FerPlus", "Caers", "CheoFaMo", "4DME", "4DME_FLOW", "4DME_FLOW_CNN"])
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--class_names", type=str, default=None,
                        help="Comma-separated class names theo đúng thứ tự label index, "
                             "vd: 'Negative,Surprise,Positive,Others'. Mặc định dùng '0','1',...")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_type", type=str, default="large", 
                    choices=["small", "base", "large"])
    parser.add_argument("--x3d_dim", type=int, default=None,
                        help="Chiều vector prior 3D. None -> tự suy ra theo data_type "
                             "(358 cho '4DME'/SMIRK, 16 cho '4DME_FLOW').")
    parser.add_argument("--x3d_hidden_dim", type=int, default=None,
                        help="Hidden dim của ThreeDMMEncoder. None -> 512 (SMIRK) hoặc "
                             "64 (flow) tuỳ theo x3d_dim được suy ra. Chỉ áp dụng khi "
                             "x3d_mode='mlp'.")
    parser.add_argument("--x3d_mode", type=str, default="mlp", choices=["mlp", "cnn"],
                        help="'mlp': prior là vector đã pool (SMIRK 358-dim hoặc "
                             "flow-pooled-vector 16-dim). 'cnn': prior là flow map thô "
                             "chưa pool [n_roi, C, H, W], dùng ThreeDMMEncoderCNN.")
    parser.add_argument("--x3d_channels", type=int, default=3,
                        help="Số kênh input cho CNN encoder (vd 3 cho u,v,optical-strain). "
                             "Chỉ áp dụng khi x3d_mode='cnn'.")

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


def save_checkpoint(state, resume_path, backup_dir, is_periodic=False, epoch=None):
    torch.save(state, resume_path)

    if is_periodic:
        for f in os.listdir(backup_dir):
            if f.startswith("4dme_epoch") and f.endswith(".pth"):
                os.remove(os.path.join(backup_dir, f))
        backup_name = f"4dme_epoch{epoch}.pth"
    else:
        backup_name = "4dme_best.pth"

    shutil.copy(resume_path, os.path.join(backup_dir, backup_name))




def main():
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.resume_dir, exist_ok=True)
    os.makedirs(args.backup_dir, exist_ok=True)
    resume_path = os.path.join(args.resume_dir, args.resume_name)

    # ---- Weights & Biases setup ----
    use_wandb = args.use_wandb
    wandb_run_id = args.wandb_run_id 

    if use_wandb:
        run_name = args.wandb_run_name or f"{args.data_type}_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=wandb_run_id, 
            resume="allow",
            mode=args.wandb_mode,
            config=vars(args),
        )
        # add epoch as a global step metric for better visualization in wandb
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    log_f = None
    if args.log_file is not None:
        os.makedirs("log", exist_ok=True)
        log_path = os.path.join("log", args.log_file)
        log_f = open(log_path, "a")
        log_f.write(f"resume path / checkpoint path: {resume_path}\n")
        log_f.write(f"Batch_size: {args.batch_size}\n")
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

    train_loader, val_loader = get_dataloaders(args)

    # Prior 3D: SMIRK (358-dim, 5 keys, mlp) mặc định; flow-pooled-vector
    # (16-dim, 1 key, mlp) khi --data_type 4DME_FLOW; hoặc flow spatial map
    # (1 key, cnn) khi --data_type 4DME_FLOW_CNN. Override thủ công qua
    # --x3d_dim/--x3d_hidden_dim/--x3d_mode/--x3d_channels nếu cần.
    if args.data_type == "4DME_FLOW_CNN":
        x3d_keys = ["flow_map"]
        x3d_dim = args.x3d_dim  # không dùng ở mode cnn, giữ None cho rõ ràng
        x3d_hidden_dim = args.x3d_hidden_dim  # không dùng ở mode cnn
        # chỉ override x3d_mode nếu người dùng chưa tự set khác "mlp" mặc định
        if args.x3d_mode == "mlp":
            args.x3d_mode = "cnn"
    elif args.data_type == "4DME_FLOW":
        x3d_keys = ["flow"]
        x3d_dim = args.x3d_dim if args.x3d_dim is not None else 16
        x3d_hidden_dim = args.x3d_hidden_dim if args.x3d_hidden_dim is not None else 64
    else:
        x3d_keys = ["exp", "jaw", "eyelid", "pose", "shape"]
        x3d_dim = args.x3d_dim if args.x3d_dim is not None else 358
        x3d_hidden_dim = args.x3d_hidden_dim  # None -> mặc định 512 trong ThreeDMMEncoder

    model = MA3D(num_classes=args.num_classes, type=args.model_type,
                  x3d_dim=x3d_dim, x3d_hidden_dim=x3d_hidden_dim,
                  x3d_mode=args.x3d_mode, x3d_channels=args.x3d_channels).to(device)

    if use_wandb and args.wandb_watch_model:
        wandb.watch(model, log="all", log_freq=100)

    CE_criterion  = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    MA_criterion = MarginAwareCELoss().to(device)

    base_optimizer = optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer,
                    lr=args.lr, weight_decay=args.weight_decay,
                    rho=0.5, adaptive=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    start_epoch = 0
    best_val_acc = 0.0
    best_val_uf1 = 0.0

    if args.resume and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_uf1 = checkpoint.get("best_val_uf1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}, best_val_uf1={best_val_uf1:.4f}")
        if wandb_run_id:
            print(f"wandb run_id: {wandb_run_id}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_loss, train_acc, train_labels, train_preds = train_one_epoch(
            model, train_loader, CE_criterion, lsce_criterion,
            MA_criterion, optimizer, device, epoch, args.epochs,
            x3d_keys=x3d_keys
        )

        val_loss, val_acc, val_labels, val_preds = validate(
            model, val_loader, CE_criterion, device, epoch, args.epochs,
            x3d_keys=x3d_keys
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

        # Use UF1 (macro-F1) for the "best" metric, acc is not good for unbalanced datasets, 
        # especially when the majority class dominates. Still track best_val_acc in parallel.
        is_best = val_uf1 > best_val_uf1

        state = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": max(best_val_acc, val_acc),
            "best_val_uf1": best_val_uf1 if not is_best else val_uf1,
        }

        best_val_acc = max(best_val_acc, val_acc)

        #  save best ckpt (UF1 / macro-F1)
        if is_best:
            best_val_uf1 = val_uf1
            save_checkpoint(state, resume_path, args.backup_dir, is_periodic=False)
            print(f"[Best] epoch={epoch+1}  val_acc={val_acc*100:.2f}%  "
                  f"UF1={val_uf1:.4f}  UAR={val_uar:.4f}")

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

        #  backup each N epoch 
        if (epoch + 1) % args.backup_every == 0:
            save_checkpoint(state, resume_path, args.backup_dir,
                            is_periodic=True, epoch=epoch+1)
            print(f"[Backup] epoch={epoch+1} → {args.backup_dir}/4dme_epoch{epoch+1}.pth")

        #  wandb log 
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

    print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%  |  Best UF1 (macro-F1): {best_val_uf1:.4f}")
    if log_f:
        log_f.write(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%  |  Best UF1: {best_val_uf1:.4f}")
        log_f.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()