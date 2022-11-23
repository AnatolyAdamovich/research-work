"""How we can implement custom optimizer in pytorch?

 1. We need to inherit from `torch.optim.Optimizer`

 2.  The constructor has 2 arguments: params and defaults
 params - an iterable (usually list of tensors)
 defaults - a dictionary mapping parameter names to their default values

 3. In the __init__ method we specify state and params_group.
 State is useful when you want to save some information
  (e.g. params from previous step in momentum)

4. zero_grad method: set all gradients equal to zero

5. step method - it is unique for all optimizers
"""
import torch
from torch.optim import Optimizer


class CustomOptimizer(Optimizer):
    """optimizer uses logic from grad file"""
    def __init__(self, params, lr=1e-3):
        # parameters validation
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        defaults = {"lr": lr}
        super(CustomOptimizer, self).__init__(params, defaults)
        print(f'self.param_groups is {self.param_groups}')


    def step(self, X_batch, y_batch, closure=None):
        """parameters updates for the custom optimizer"""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        batch_len = len(X_batch)
        for group in self.param_groups:
            lr = group['lr']
            for p in group["params"]:

                params_len = len(p[0])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                state['step'] += 1

                if params_len != batch_len:
                    # 1st case
                    k = batch_len // params_len
                    X_batch_patches = X_batch.reshape(k, params_len, *list(X_batch.shape[1:]))
                    y_batch_patches = y_batch.reshape(k, params_len, *list(y_batch.shape[1:]))
                    print(f'Patches shape: {X_batch_patches.shape}')
                    estimates = []
                    for i in range(k):
                        det = torch.det(X_batch_patches[i])
                        inv = torch.linalg.inv(X_batch_patches[i])
                        adj = det * inv
                        step_size = lr / (1 + lr * det**2)

                        params_estimation = step_size * det * (adj @ y_batch_patches[i] - det * p.reshape(params_len, -1))
                        estimates.append(params_estimation)
                    estimates = torch.stack(estimates)
                    with torch.no_grad():
                        p += torch.mean(estimates, dim=0).reshape(len(p), -1)
                else:
                    # 2nd case
                    det = torch.det(X_batch)
                    inv = torch.linalg.inv(X_batch)
                    adj = det * inv
                    step_size = (lr * det) / (1 + lr * det**2)
                    with torch.no_grad():
                        p += step_size * (adj @ y_batch - det * p)

                # the next layer require another size of input tensor
                X_batch = X_batch @ p.reshape(params_len, -1)
        return loss


class DREMOptimizer(Optimizer):
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        defaults = {"lr": lr}
        super(DREMOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, determinant_X_batch, closure=None):
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

                state["step"] = state.get("step", 0) + 1
                gradient.div_(1 + lr * determinant_X_batch**2)
                p.data.add_(gradient, alpha=-lr)

        return loss
