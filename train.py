import os
import torch
import torch.optim as optim
from models.MA3D import MA3D
from loss_function.loss import MarginAwareCELoss, LabelSmoothingCrossEntropy
from models.sam import SAM

from build_dataloader import get_dataloaders
from engine import train_one_epoch, validate
import time
import argparse

import wandb




def get_args():
    parser = argparse.ArgumentParser("SFER Training")

    # Dataset
    parser.add_argument("--data_type", default="RAF-DB", choices=["RAF-DB", "VKIST", "Cheo", "FerPlus", "Caers", "CheoFaMo", "4DME"])
    parser.add_argument("--num_classes", type=int, default=7)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)

    # Logging
    parser.add_argument("--log_file", type=str, default="log.txt")

    # Checkpoint / Resume
    parser.add_argument("--resume_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_name", type=str, default="last.pth")
    parser.add_argument("--resume", action="store_true")

    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true", help="turn on wandb logging")
    parser.add_argument("--wandb_project", type=str, default="MA3D-micro", help="project name in wandb")
    parser.add_argument("--wandb_entity", type=str, default=None, help="team/entity name in wandb, or default")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="run name in wandb")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_watch_model", action="store_true", help="Log gradients/weights histogram")

    return parser.parse_args()

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.resume_dir, exist_ok=True)
    resume_path = os.path.join(args.resume_dir, args.resume_name)

    # ---- Weights & Biases setup ----
    use_wandb = args.use_wandb
    if use_wandb:
        run_name = args.wandb_run_name or f"{args.data_type}_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
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


    train_loader, val_loader = get_dataloaders(args)

    model = MA3D(num_classes= args.num_classes).to(device)

    if use_wandb and args.wandb_watch_model:
        wandb.watch(model, log="all", log_freq=100)

    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
    MA_criterion = MarginAwareCELoss().to(device)

    base_optimizer = optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay, rho=0.5, adaptive=True,)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, CE_criterion, lsce_criterion, MA_criterion, optimizer, device, epoch, args.epochs
        )

        val_loss, val_acc = validate(
            model, val_loader, CE_criterion, device, epoch, args.epochs
        )

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        epoch_time = (time.time() - epoch_start_time) / 60.0

        is_best = val_acc > best_val_acc

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "lr": lr,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "epoch_time_min": epoch_time,
                "best_val_acc": best_val_acc if not is_best else val_acc,
            })

        if log_f is not None:
            log_f.write(
                f"{epoch + 1:^6d} {lr:^12.8f} {train_loss:^12.4f} {train_acc * 100:^10.2f} "
                f"{val_loss:^12.4f} {val_acc * 100:^10.2f} {epoch_time:^10.2f}\n"
            )
            log_f.flush()

        if is_best:
            print(f"epoch: {epoch + 1}, val_acc: {val_acc * 100:.2f}%")
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
            }, resume_path)

            if use_wandb:
                wandb.run.summary["best_val_acc"] = best_val_acc
                wandb.run.summary["best_epoch"] = epoch + 1

            if log_f is not None:
                log_f.write(
                    f"BEST\tval_acc={val_acc * 100:.2f}\n"
                )
                log_f.flush()

    print(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%")
    if log_f is not None:
        log_f.write(f"\nBest validation accuracy: {best_val_acc * 100:.2f}%")
        log_f.close()

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()