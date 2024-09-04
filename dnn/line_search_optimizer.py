import torch
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy

class LS(Optimizer):

    def __init__(self, params, lr=1.0, max_step_size=1.0, slope_modifier=1e-4, shrink_factor=0.5):
        defaults = dict(lr=lr, max_step_size=max_step_size, slope_modifier=slope_modifier, shrink_factor=shrink_factor)
        super(LS, self).__init__(params, defaults)
        self._params = self.param_groups[0]['params']

    def __setstate__(self, state):
        super(LS, self).__setstate__(state)

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)        

    def _copy_params(self):
        current_params = []
        for param in self._params:
            current_params.append(deepcopy(param.data))
        return current_params

    def _load_params(self, current_params):
        i = 0
        for param in self._params:
            param.data[:] = current_params[i]
            i += 1

    def _add_update(self, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(update[offset:offset + numel].view_as(p.data))
            offset += numel
        
    def step(self, closure=required):
        group = self.param_groups[0]
        max_step_size = group['max_step_size']
        slope_modifier = group['slope_modifier']
        shrink_factor = group['shrink_factor']
        eps = 1e-8

        fc = closure()

        fp = self._gather_flat_grad()
        fp_l2 = torch.linalg.norm(fp)

        delta = -fp / (fp_l2 + eps) * max_step_size
        delta_l2 = max_step_size
        self._add_update(delta)
        fn = closure()

        while fn > fc - slope_modifier * fp_l2 * delta_l2:
            delta.mul_(shrink_factor)
            delta_l2 = delta_l2 * shrink_factor
            self._add_update(-delta)
            fn = closure()
        
        return fn