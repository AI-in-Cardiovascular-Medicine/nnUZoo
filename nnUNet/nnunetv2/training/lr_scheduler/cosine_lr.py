import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        lr_end: float = 1e-6, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        lr_end (`float`, *optional*, defaults to 1e-6):
                The end LR.
        last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.
        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        lr_range = lr_init - lr_end
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr_new = lr_end + lr_range * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        lr_new /= lr_init
        return max(0.0, lr_new)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

if __name__ == '__main__':
    from torch.optim import Adam
    from timm.models import InceptionResnetV2

    model = InceptionResnetV2()
    lr_scheduler = get_cosine_schedule_with_warmup(
        Adam(model.parameters()),
        num_warmup_steps=500,
        num_training_steps=50000,
    )
    data = []
    for i in range(50000):
        lr_scheduler.step()
        data.append(lr_scheduler._last_lr[0])
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()