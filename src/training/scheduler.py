import math

class CosineLRScheduler:
    def __init__(self, optimizer, max_epochs: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.epoch = 0
        
    def step(self):
        self.epoch += 1
        self._update_lrs()
        
    def _update_lrs(self):
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                1 + math.cos(math.pi * self.epoch / self.max_epochs)
            )
            group["lr"] = lr
    
    def state_dict(self):
        return {"epoch": self.epoch}
    
    def load_state_dict(self, state):
        self.epoch = state["epoch"]
        self._update_lrs()