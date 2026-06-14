import os
import torch
from torch.cuda.amp import GradScaler
import yaml

from src.dataset import MSADataset
from src.tokenizer import AlignTokenizer, GapTokenizer
from src.model.msa_transformer import MSATransformer
from src.training.loss import MSALoss
from src.training.train import train_one_epoch, evaluate
from src.training.scheduler import CosineLRScheduler
from src.training.utils import save_checkpoint, load_checkpoint, load_model_only, get_incremental_path

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(BASE_DIR, "configs", "model.yaml")) as f:
        model_cfg = yaml.safe_load(f)["model"]
        
    with open(os.path.join(BASE_DIR, "configs", "train.yaml")) as f:
        train_cfg = yaml.safe_load(f)
        
    stage_cfg = train_cfg["stage"]
    seq_count = stage_cfg["seq_count"]

    resume_cfg = train_cfg.get("resume", {})
    resume_enable = resume_cfg.get("enable", False)
    resume_mid_epoch = resume_cfg.get("resume_mid_epoch", False)
    load_prev_stage = resume_cfg.get("load_prev_stage", False)
    
    if resume_mid_epoch and load_prev_stage:
        raise ValueError(
            "resume_mid_epoch and load_prev_stage cannot both be true."
        )
    
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    start_epoch = 0
    start_step = 0
        
    align = AlignTokenizer()
    gap = GapTokenizer()
    
    model = MSATransformer(
        align_vocab_size=len(align.vocab),
        gap_vocab_size=len(gap.vocab),
        dim=model_cfg["dim"],
        enc_depth=model_cfg["enc_depth"],
        dec_depth=model_cfg["dec_depth"],
        n_heads=model_cfg["n_heads"],
        ff_dim=model_cfg["ff_dim"],
        dropout=model_cfg["dropout"],
        max_len=model_cfg["max_len"],
        align_pad_id=align.pad_id,
        gap_pad_id=gap.pad_id,
    ).to(device)
    
    print(f"[DEVICE] Model on : {next(model.parameters()).device}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_cfg["lr"])
    )
    
    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        max_epochs=stage_cfg["epochs"],
        min_lr=0.0
    )

    ckpt_dir = os.path.join(BASE_DIR, "experiments", "ckpt",f"stage-{seq_count}")
    stg_dir = os.path.join(BASE_DIR, "experiments", "final")
    
    ckpt_model = os.path.join(BASE_DIR, "ckpt", f"stage-{seq_count}")
    os.makedirs(ckpt_model, exist_ok=True)
    
    if resume_enable and resume_mid_epoch:
        ckpt_path = os.path.join(ckpt_dir, f"stage-{seq_count}-ckpt.pt")
        print(f"[INFO] Resume training from checkpoint: {ckpt_path}")
        start_epoch = load_checkpoint(
            model, optimizer, scaler, scheduler, ckpt_path
        )

    elif load_prev_stage and seq_count > 2:
        prev_path = os.path.join(stg_dir, f"stage-{seq_count-1}.pt")
        print(f"[INFO] Load previous stage model: {prev_path}")
        load_model_only(model, prev_path)

    else:
        print("[INFO] Training from scratch")
    
    DATA = os.path.join(BASE_DIR, f"data/msa-nuc-{seq_count}")
    
    train_dataset = MSADataset(
        path=os.path.join(DATA, "train.jsonl"),
        max_len=model_cfg["max_len"],
        max_samples=train_cfg["data"]["max_samples"]
    )

    val_dataset = MSADataset(
        path=os.path.join(DATA, "val.jsonl"),
        max_len=model_cfg["max_len"]
    )

    test_dataset = MSADataset(
        path=os.path.join(DATA, "test.jsonl"),
        max_len=model_cfg["max_len"]
    )
    
    print(f"[INFO] Train size: {len(train_dataset)} samples")
    print(f"[INFO] Val size: {len(val_dataset)} samples")
    print(f"[INFO] Test size: {len(test_dataset)} samples")
    
    
    loss_fn = MSALoss(align_pad_id=align.pad_id, gap_pad_id=gap.pad_id)
    
    for epoch in range(start_epoch, stage_cfg["epochs"]):
        print(f"\n[Stage {seq_count}] Epoch {epoch+1}/{stage_cfg['epochs']}")
        
        train_loss = train_one_epoch(
            model=model,
            dataset=train_dataset,
            loss_fn=loss_fn,
            optimizer=optimizer,
            batch_size=stage_cfg["batch_size"],
            scaler=scaler,
            target_batch=64
        )
        
        val_loss = evaluate(
            model=model,
            dataset=val_dataset,
            loss_fn=loss_fn,
            batch_size=stage_cfg["batch_size"]
        )
        
        print(
            f"[epoch-end] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        
        scheduler.step()
            
        resume_path = os.path.join(ckpt_dir, f"stage-{seq_count}-ckpt.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch+1,
            path=resume_path,
            model_cfg=model_cfg
        )
            
    test_loss = evaluate(
        model=model,
        dataset=test_dataset,
        loss_fn=loss_fn,
        batch_size=stage_cfg["batch_size"]
    )
    print(f"[TEST] avg_loss={test_loss:.4f}")
        
    save_path = os.path.join(
        BASE_DIR, "experiments", "epoch-50", f"stage-{seq_count}.pt"
    )
    
    save_path = get_incremental_path(save_path)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        epoch=stage_cfg["epochs"],
        path=save_path,
        model_cfg=model_cfg
    )

    
    print(f"[OK] Saved: {save_path}")
    
if __name__ == "__main__":
    main()
