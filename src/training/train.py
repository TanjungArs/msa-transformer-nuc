import torch
import time
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader


def train_one_step(
    model,
    batch,
    loss_fn,
    optimizer,
    grad_clip_norm: float = 1.0,
    scaler: GradScaler | None = None,
    accum_steps: int = 1,
    global_steps: int = 0
):
    model.train()
    device = next(model.parameters()).device
    batch = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    
    use_amp = scaler is not None and torch.cuda.is_available()
    
    with autocast(enabled=use_amp):
        logits_align, logits_gap = model(
            input_ids=batch["input_ids"],
            key_padding_mask=batch["key_padding_mask"],
            decoder_input_align=batch["decoder_input_align"],
            decoder_input_gap=batch["decoder_input_gap"],
        )
        
        loss, align_loss, gap_loss = loss_fn(
            logits_align, batch["labels_align"],
            logits_gap, batch["labels_gap"]
        )
    
    loss = loss / accum_steps

    if not torch.isfinite(loss):
        print("[Error] NaN / Inf loss detected. stop training")
        return None
       
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    do_step = ((global_steps + 1) % accum_steps == 0)
    
    if do_step:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip_norm
            )
            optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)            
        
    return loss.item() * accum_steps

def train_one_epoch(
    model,
    dataset,
    loss_fn,
    optimizer,
    batch_size: int,
    shuffle: bool = True,
    scaler: GradScaler | None = None,
    target_batch: int | None = None
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
        num_workers=2,
        pin_memory=True
    )

    total_loss = 0.0
    steps = 0
    global_steps = 0
    start_time = time.time()

    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(loader):
        
        actual_bs = batch["input_ids"].size(0)
        
        if target_batch is not None:
            accum_steps = max(1, target_batch // actual_bs)
        else:
            accum_steps = 1
        
        loss = train_one_step(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            accum_steps=accum_steps,
            global_steps=global_steps
        )
        
        if loss is None:
            return None
        
        total_loss += loss
        steps += 1
        global_steps += 1

        if (step + 1) % max(1, len(loader) // 4) == 0:
            elapsed = (time.time() - start_time) / 60
            print(
                f"[train] {100 * (step + 1) / len(loader):5.1f}% "
                f"| elapsed={elapsed:.1f} min "
                f"| avg_loss={total_loss / steps:.4f}"
            )

    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(
    model,
    dataset,
    loss_fn,
    batch_size: int
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn
    )

    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    steps = 0
    
    for batch in loader:
        
        batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        
        logits_align, logits_gap = model(
            input_ids=batch["input_ids"],
            key_padding_mask=batch["key_padding_mask"],
            decoder_input_align=batch["decoder_input_align"],
            decoder_input_gap=batch["decoder_input_gap"]
        )
        
        loss, _, _ = loss_fn(
            logits_align, batch["labels_align"],
            logits_gap, batch["labels_gap"]
        )

        total_loss += loss.item()
        steps += 1
        
    return total_loss / max(steps, 1)