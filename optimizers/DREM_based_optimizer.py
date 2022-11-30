"""This script contains an implementation of the optimizer
which is based on the DREM procedure
More info can be found in the `report` folder
"""


import torch
from torch.optim import Optimizer


class DREMOptimizer(Optimizer):
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        defaults = {"lr": lr}
        super(DREMOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, det_batch, partition=0, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                gradient = p.grad.data
                state = self.state[p]
                if partition == 0:
                    # case 1: batch size == number of input feature (in the 1st layer)
                    state["step"] = state.get("step", 0) + 1
                    gradient.div_(1 + lr * (det_batch ** 2))
                    p.data.add_(gradient, alpha=-lr)
                else:
                    # case 2: batch size == s * n, where n == number of input feature (in the 1st layer)
                    state['part'] = state.get("part", 0) + 1
                    gradient.div_(1 + lr * (det_batch ** 2))
                    state["cum_grad"] = state.get("cum_grad", 0.0) + (lr * gradient)
                    if partition == 1:
                        mean_cum_grad = state["cum_grad"] / state["part"]
                        p.data.add_(mean_cum_grad, alpha=-1)
        return loss
