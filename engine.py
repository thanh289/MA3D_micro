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


def prepare_batch(batch, device):
    apex = batch["apex"].to(device, non_blocking=True)
    onset = batch["onset"].to(device, non_blocking=True)
    flow = batch["flow"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return apex, onset, flow, labels


def train_one_epoch(model, loader, CE_criterion, lsce_criterion, MA_criterion,
                    optimizer, device, epoch, epochs):

    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for batch_idx, batch in enumerate(
        tqdm(loader, desc=f"Training [{epoch + 1}/{epochs}]", leave=False)
    ):
        apex, onset, flow, labels = prepare_batch(batch, device)

        logits, features, attn = model(apex, onset, flow)
        loss = get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.first_step(zero_grad=True)

        logits_2, features_2, attn = model(apex, onset, flow)
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
def validate(model, loader, criterion, device, epoch, epochs):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for batch in tqdm(loader, desc=f"Validation [{epoch + 1}/{epochs}]", leave=False):
        apex, onset, flow, labels = prepare_batch(batch, device)

        logits, features, attn = model(apex, onset, flow)
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