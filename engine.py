import torch
from tqdm import tqdm


def get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch):
    CE_loss = CE_criterion(logits, labels)
    lsce_loss = lsce_criterion(logits, labels)
    MA_loss = MA_criterion(logits, labels)

    if epoch < 10:
        return 2 * lsce_loss + CE_loss  # warm up
    else:
        return MA_loss 

DEFAULT_X3D_KEYS = ["exp", "jaw", "eyelid", "pose", "shape"]  # SMIRK, tổng 358-dim


def prepare_batch(batch, device, x3d_keys=None):
    """
    x3d_keys: thứ tự các key trong batch["npy"] sẽ được ghép nối thành x_3d.
        - None / mặc định: DEFAULT_X3D_KEYS (SMIRK, 358-dim, x3d_mode="mlp")
        - ["flow"]: prior mới từ optical flow. Có thể là:
            * vector đã pool (16-dim)          -> MA3D(x3d_mode="mlp")
            * map thô [3, 42, 42] chưa pool -> MA3D(x3d_mode="mean")
          Với 1 key duy nhất, torch.cat(dim=1) chỉ là pass-through nên tensor
          giữ nguyên shape/rank (2D hay 5D đều đi qua được) -- không cần đổi
          gì ở đây khi chuyển giữa 2 mode, chỉ cần model (MA3D) khớp mode.
    Thứ tự ghép PHẢI cố định giữa lúc train và lúc load checkpoint cũ, nên không
    dùng sorted(keys) tự động mà dùng đúng list truyền vào / mặc định.
    """
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)

    keys = x3d_keys if x3d_keys is not None else DEFAULT_X3D_KEYS
    npy = batch["npy"]
    parts = [npy[k].to(device, non_blocking=True) for k in keys]

    x_3d = torch.cat(parts, dim=1)  # [B, sum(dims)]  -- 358 (SMIRK) hoặc 16 (flow)

    return images, labels, x_3d

def train_one_epoch(model, loader, CE_criterion, lsce_criterion, MA_criterion,
                    optimizer, device, epoch, epochs, x3d_keys=None):

    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []   

    for batch_idx, batch in enumerate(
        tqdm(loader, desc=f"Training [{epoch + 1}/{epochs}]", leave=False)
    ):
        images, labels, x_3d = prepare_batch(batch, device, x3d_keys=x3d_keys)

        logits, features, attn = model(images, x_3d)
        loss = get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        logits_2, features_2, attn = model(images, x_3d)
        loss_2 = get_loss(logits_2, labels, CE_criterion, lsce_criterion, MA_criterion, epoch)
        loss_2.backward()
        optimizer.second_step(zero_grad=True)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size

        all_labels.append(labels.detach().cpu())    
        all_preds.append(preds.detach().cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()  

    return epoch_loss, epoch_acc, all_labels, all_preds   


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, epochs, x3d_keys=None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for batch in tqdm(loader, desc=f"Validation [{epoch + 1}/{epochs}]", leave=False):
        images, labels, x_3d = prepare_batch(batch, device, x3d_keys=x3d_keys)

        logits, features, attn = model(images, x_3d)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    return epoch_loss, epoch_acc, all_labels, all_preds