import torch

class SGD:
    # Gradients are accumulated into the momentum buffer as they are estimated; apply() moves the weights once per step.
    def __init__(self, groups, lr, momentum=0.0):
        self.lr = {p: lr[name] if isinstance(lr, dict) else lr for name, params in groups.items() for p in params}
        self.momentum, self.buffers, self.accumulated = momentum, {}, set()

    def accumulate(self, param, grad):
        grad = grad.to(param.dtype)
        if param in self.accumulated:
            self.buffers[param] += grad
        elif param in self.buffers:
            self.buffers[param].mul_(self.momentum).add_(grad)
        else:
            self.buffers[param] = grad.clone()
        self.accumulated.add(param)

    @torch.no_grad()
    def apply(self):
        for param in self.accumulated:
            param -= self.buffers[param] * self.lr[param]
        self.accumulated.clear()
