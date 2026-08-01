import torch
from tqdm import tqdm


def get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch,
             aux=None, aux_loss_weight=0.2):
    """
    aux: the `aux` dict returned by MA3D.forward() -- {"rise": logits or
        None, "fall": logits or None, "agreement": ... or None}.

        use_rise_fall=False: aux["rise"]/aux["fall"] both None -> no aux
            loss, reduces to exactly the old main-loss-only behaviour.
        rise_fall_mode="feature_gate": BOTH aux["rise"] and aux["fall"]
            are populated (small aux heads, no gradient path except
            through this term) -> aux_loss = CE(rise) + CE(fall).
        rise_fall_mode="decision_level": ONLY aux["fall"] is populated
            (aux["rise"] is deliberately None -- `logits`/`out` IS
            out_rise already, so its CE is already counted via the main
            loss above; adding it again via aux["rise"] would double-count
            the exact same tensor, which GAMDSS's real training script
            does NOT do -- see MA3D.py class docstring) -> aux_loss =
            CE(fall) only, matching GAMDSS's `loss = CE(ALL,y) + CE(s,y)`.

        Either term is summed independently (not requiring both to be
        present), so this one function correctly covers all 3 cases above.
    aux_loss_weight: weight applied to the aux CE term(s) -- kept separate
        from the main loss's own weighting (warm-up vs. MACE) below, and
        always plain CE regardless of epoch (no warm-up/MACE staging
        needed for the auxiliary terms -- MACE targets easy/hard SAMPLE
        weighting via margin, which is a main-loss-level concern here).
        NOTE: GAMDSS's real script sums CE(ALL)+CE(s) with an IMPLICIT
        weight of 1.0 (no scaling at all) for rise_fall_mode=
        "decision_level" -- the default aux_loss_weight=0.2 here is a
        DELIBERATE deviation (lighter aux signal); try 1.0 to match
        GAMDSS's real weighting exactly, and compare.
    """
    CE_loss = CE_criterion(logits, labels)
    lsce_loss = lsce_criterion(logits, labels)
    MA_loss = MA_criterion(logits, labels)

    if epoch < 10:
        main_loss = 2 * lsce_loss + CE_loss  # warm up
    else:
        main_loss = MA_loss

    aux_loss = 0.0
    if aux is not None:
        if aux.get("rise") is not None:
            aux_loss = aux_loss + CE_criterion(aux["rise"], labels)
        if aux.get("fall") is not None:
            aux_loss = aux_loss + CE_criterion(aux["fall"], labels)

    return main_loss + aux_loss_weight * aux_loss


def prepare_batch(batch, device):
    apex = batch["apex"].to(device, non_blocking=True)
    onset = batch["onset"].to(device, non_blocking=True)
    flow_rise = batch["flow_rise"].to(device, non_blocking=True)
    flow_fall = batch["flow_fall"].to(device, non_blocking=True) if "flow_fall" in batch else None
    offset = batch["offset"].to(device, non_blocking=True) if "offset" in batch else None
    au = batch["au"].to(device, non_blocking=True) if "au" in batch else None
    labels = batch["label"].to(device, non_blocking=True)
    return apex, onset, flow_rise, flow_fall, offset, au, labels


def train_one_epoch(model, loader, CE_criterion, lsce_criterion, MA_criterion,
                    optimizer, device, epoch, epochs, aux_loss_weight=0.2):

    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for batch_idx, batch in enumerate(
        tqdm(loader, desc=f"Training [{epoch + 1}/{epochs}]", leave=False)
    ):
        apex, onset, flow_rise, flow_fall, offset, au, labels = prepare_batch(batch, device)

        logits, features, attn, aux = model(apex, onset, flow_rise, flow_fall, offset, au)
        loss = get_loss(logits, labels, CE_criterion, lsce_criterion, MA_criterion, epoch,
                         aux=aux, aux_loss_weight=aux_loss_weight)

        optimizer.zero_grad()
        loss.backward()
        # optimizer.first_step(zero_grad=True)
        optimizer.step()

        # logits_2, features_2, attn, aux_2 = model(apex, onset, flow_rise, flow_fall, offset, au)
        # loss_2 = get_loss(logits_2, labels, CE_criterion, lsce_criterion, MA_criterion, epoch,
        #                    aux=aux_2, aux_loss_weight=aux_loss_weight)
        # loss_2.backward()
        # optimizer.second_step(zero_grad=True)

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
    """
    Val loss is MAIN-loss-only (criterion applied to `logits`, exactly as
    before) -- the auxiliary rise/fall heads are a TRAINING-time regularizer
    (extra gradient signal), not part of what we're actually trying to
    minimize on held-out data, so they're intentionally excluded here (aux
    is still returned by model(...) but simply unused).
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for batch in tqdm(loader, desc=f"Validation [{epoch + 1}/{epochs}]", leave=False):
        apex, onset, flow_rise, flow_fall, offset, au, labels = prepare_batch(batch, device)

        logits, features, attn, aux = model(apex, onset, flow_rise, flow_fall, offset, au)
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