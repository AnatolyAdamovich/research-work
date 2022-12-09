"""
This script contains an implementation of the optimizer
that uses a two-phase approach,
where the DREM operator is used in the 1st phase
and finite time estimation is used in the 2nd phase

More info can be found in the `report` folder
"""

import torch
from torch.optim import Optimizer
from scipy.interpolate import interp1d
from scipy.integrate import simpson, trapezoid


class FiniteTimeOptimizer(Optimizer):
    def __init__(self, params, lr, n_of_batches):

        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if (n_of_batches <= 0) or type(n_of_batches) != int:
            raise ValueError(f"Invalid number of batches: {n_of_batches} - should be >= 1 and should have `int` type")
        defaults = {"lr": lr, "number_of_batches": n_of_batches}

        super(FiniteTimeOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, det_batch, partition=0, t=2, closure=None):
        """
        single optimization step
        Parameters:
            det_batch (torch.tensor or float): determinant of batch
            partition (int): batch's part (for the second case, when batch has the size s*n)
            t (int): batch's number (from 1 to self.state['number_of_batches']) / each `t` batches apply finite time
            closure (callable): functionality to recalculate loss function // see torch.optim docs for more
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            n_of_batches = group['number_of_batches']
            for p in group["params"]:
                if p.grad is None:
                    continue
                gradient = p.grad.data
                state = self.state[p]
                if "step" not in state:
                    state['step'] = 0
                    state['initialization'] = p
                if t % n_of_batches == 0:
                    self._finite_time(state, lr, p)
                    print('applied finite time optimizer')
                    self._drem_operator(det_batch, partition, state, lr, gradient, p)
                else:
                    self._drem_operator(det_batch, partition, state, lr, gradient, p)

        return loss

    def _drem_operator(self, det_batch, partition, state, lr, gradient, p):
        """Functional API that performs optimizers computations"""
        if "determinants" in state:
            state["determinants"].append(det_batch)
        else:
            state["determinants"] = [det_batch]
        if partition == 0:
            # case 1: batch size == number of input feature (in the 1st layer)
            state["step"] += state.get("step", 0) + 1
            gradient.div_(1 + lr * (det_batch ** 2))
            p.data.add_(gradient, alpha=-lr)
        else:
            # case 2: batch size == s * n, where n == number of input feature (in the 1st layer)
            state['part'] = state.get("part", 0) + 1
            gradient.div_(1 + lr * (det_batch ** 2))
            state["cum_grad"] = state.get("cum_grad", 0.0) + (lr * gradient)
            if partition == 1:
                mean_cum_grad = state["cum_grad"] / state["part"]
                state['part'] = 0
                state['step'] += 1
                p.data.add_(mean_cum_grad, alpha=-1)

    def _finite_time(self, state, lr, p):
        """Functional API that performs finite-time algorithm computations"""
        x_range = torch.arange(len(state['determinants']))
        squared_determinants = torch.tensor(state["determinants"])**2
        integral = simpson(y=squared_determinants,
                           x=x_range)
        # determinant_function = interp1d(x, state['determinants'],
        #                                 fill_value='extrapolate')

        # def square_det(a):
        #     return determinant_function(a)**2
        #
        # integral, error = quad(func=square_det,
        #                        a=0,
        #                        b=n_of_batches)

        w = torch.exp(torch.ones_like(p) * (-lr) * integral)
        p = (p - state["initialization"] * w) / (1 - w)
