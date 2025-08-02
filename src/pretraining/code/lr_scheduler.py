from torch.optim.lr_scheduler import _LRScheduler
import math

class CustomCosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, max_iters, warmup_iters=None, min_lr_ratio=0.1, last_epoch=-1):
        self.max_lr = optimizer.param_groups[0]['lr']  # Get initial learning rate
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters if warmup_iters is not None else int(max_iters * 0.1)
        self.min_lr = self.max_lr * min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        it = self.last_epoch  # Current iteration
        
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return [self.max_lr * (it + 1) / self.warmup_iters for _ in self.base_lrs]
        
        # 2) if it > max_iters, return min learning rate
        if it > self.max_iters:
            return [self.min_lr for _ in self.base_lrs]
        
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.max_iters - self.warmup_iters)
        decay_ratio = min(max(0.0, decay_ratio), 1.0)  # Ensure between 0 and 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return [self.min_lr + coeff * (self.max_lr - self.min_lr) for _ in self.base_lrs]