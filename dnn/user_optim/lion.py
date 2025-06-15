import torch
from torch.optim.optimizer import Optimizer, required

class Lion(Optimizer):
    def __init__(self, params, lr=required, beta1=0.9, beta2=0.99, gamma=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, gamma=gamma)
        super(Lion, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Lion, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            gamma = group['gamma']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if d_p.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    m = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    m = param_state['momentum_buffer']

                c = beta1 * m + (1 - beta1) * d_p
                m.mul_(beta2).add_(d_p, alpha=1 - beta2)
                delta = torch.sign(c).add_(p, alpha=gamma)

                p.add_(delta, alpha=-lr)

        return loss