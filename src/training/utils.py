import os
import torch

def save_checkpoint(model, optimizer, scaler, scheduler, epoch: int, path:str, model_cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "model_cfg": model_cfg
        },
        path
    )
    
def load_checkpoint(model, optimizer, scaler, scheduler, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"]

    model.load_state_dict(state, strict=False)

    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
    except Exception as e:
        print(f"Optimizer/Scaler tidak dimuat karena dimensi berbeda: {e}")

    return ckpt.get("epoch", 0)


def load_model_only(model, path:str):
    state = torch.load(path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    
def get_incremental_path(path: str):
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    i = 1
    while True:
        new_path = f"{base}({i}){ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1