import warnings
from typing import List
import math
from torch.optim.lr_scheduler import _LRScheduler
import torch

class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None,
                 verbose:bool = True):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.verbose = verbose
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

class ContinuedPolyLRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        start_epoch: int,
        initial_lr,
        warmup_lr,
        warmup_epochs,
        total_epochs,
        final_lr,
        last_epoch=-1,
        exponent: float = 0.9,
            verbose: bool = True
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.start_epoch: int = start_epoch
        self.final_lr = final_lr
        self.exponent = exponent
        self.ctr = 0
        self.verbose = verbose
        super(ContinuedPolyLRSchedulerWithWarmup, self).__init__(optimizer, last_epoch)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < (self.warmup_epochs + self.start_epoch):
            new_lr = self.warmup_lr + (self.initial_lr - self.warmup_lr) * (
                (max(0, current_step - self.start_epoch)) / (self.warmup_epochs)
            )
        else:
            decay_steps = self.total_epochs - self.start_epoch - self.warmup_epochs
            adjusted_step = current_step - self.start_epoch - self.warmup_epochs
            new_lr = self.final_lr + (self.initial_lr - self.final_lr) * (
                (1 - (adjusted_step / decay_steps)) ** self.exponent
            )
        if self.verbose:
            print(f"[INFO] NEW_LR: {new_lr}, warmup_lr: {self.warmup_lr}, initial_lr: {self.initial_lr}, current_step: {current_step}, start_epoch: {self.start_epoch}, warmup_epoch: {self.warmup_epochs}")
        for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
        return new_lr

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

if __name__ == '__main__':
    from torch.optim import Adam
    from timm.models import InceptionResnetV2

    model = InceptionResnetV2()
    lr_scheduler = ContinuedPolyLRSchedulerWithWarmup(
        Adam(model.parameters()),
        start_epoch=0,
        initial_lr=1e-4,
        warmup_lr=1e-5,
        final_lr=1e-6,
        total_epochs=398,
        warmup_epochs=50,
    )
    data = []
    for i in range(398):
        val = lr_scheduler.step()
        print(val)
        data.append(val)
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()

    lr_scheduler = LinearWarmupCosineAnnealingLR(
        Adam(model.parameters()),
        warmup_start_lr=1e-4,
        max_epochs=400,
        warmup_epochs=50,
    )
    data = []
    for i in range(400):
        val = lr_scheduler.step()
        print(val)
        data.append(val)
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()


    lr_scheduler = PolyLRScheduler(
        Adam(model.parameters()),
        initial_lr=1e-3,
        max_steps=1000
    )
    data = []
    for i in range(398):
        val = lr_scheduler.step()
        print(val)
        data.append(val)
    import matplotlib.pyplot as plt

    plt.plot(data)
    plt.show()