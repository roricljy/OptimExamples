import torch
from torch.optim.optimizer import Optimizer, required

# Muon: MomentUm Orthogonalized by Newton-schulz
# Weights with 2 or more dimensions are updated by the orthogonalized momentum.
# 1-D parameters (bias, norm scale) cannot be orthogonalized and are updated by Adam.
class Muon(Optimizer):
    def __init__(self, params, lr=required, momentum=0.95, weight_decay=0.0, ns_steps=5, adam_lr=3e-4, adam_betas=(0.9, 0.95)):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, adam_lr=adam_lr, adam_betas=adam_betas)
        super(Muon, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Muon, self).__setstate__(state)

    # Approximates U V^T of the SVD G = U S V^T with a quintic polynomial iteration
    @staticmethod
    def newton_schulz(G, steps):
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G / (G.norm() + 1e-7)
        transposed = G.size(0) > G.size(1)
        if transposed:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transposed:
            X = X.T
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            adam_lr = group['adam_lr']
            adam_beta1, adam_beta2 = group['adam_betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if d_p.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                param_state = self.state[p]

                if p.dim() >= 2:
                    if 'momentum_buffer' not in param_state:
                        m = param_state['momentum_buffer'] = torch.zeros_like(p)
                    else:
                        m = param_state['momentum_buffer']

                    m.mul_(momentum).add_(d_p)
                    g = d_p.add(m, alpha=momentum)  # Nesterov momentum

                    # Conv kernel (out, in, kh, kw) is flattened to (out, in*kh*kw)
                    G = g.view(g.size(0), -1)
                    O = self.newton_schulz(G, ns_steps)
                    O = O * max(1.0, G.size(0) / G.size(1)) ** 0.5
                    delta = O.view_as(p).add(p, alpha=weight_decay)

                    p.add_(delta, alpha=-lr)
                else:
                    if 'step' not in param_state:
                        param_state['step'] = 0
                        param_state['exp_avg'] = torch.zeros_like(p)
                        param_state['exp_avg_sq'] = torch.zeros_like(p)
                    param_state['step'] += 1
                    t = param_state['step']
                    exp_avg = param_state['exp_avg']
                    exp_avg_sq = param_state['exp_avg_sq']

                    exp_avg.mul_(adam_beta1).add_(d_p, alpha=1 - adam_beta1)
                    exp_avg_sq.mul_(adam_beta2).addcmul_(d_p, d_p, value=1 - adam_beta2)
                    m_hat = exp_avg / (1 - adam_beta1 ** t)
                    v_hat = exp_avg_sq / (1 - adam_beta2 ** t)
                    delta = m_hat / (v_hat.sqrt() + 1e-8)

                    p.add_(delta, alpha=-adam_lr)

        return loss
