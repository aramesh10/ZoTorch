import torch

import zotorch as zt

# MeZO (https://arxiv.org/abs/2305.17333):
# Memory efficient SPSA on the weights
#
#     estimators = [MeZO(model, [name for name, _ in model.named_parameters()], eps=1e-3, n=8)]
#     for estimator in estimators:
#         estimator(model, batch, opt)
#     opt.apply()
#
# zt.seed() draws a fresh seed, and zt.noise(seed) draws the same noise from it again, so it is never stored.

def add_noise(params, seed, scale):
    for param in params:
        param += zt.noise(seed) * scale

class MeZO(zt.Estimator):
    # Every draw perturbs all of the weights, each with its own seed.
    def __init__(self, model, names, eps=1e-3, n=1):
        super().__init__([zt.ZoWeight(model, name) for name in names])
        self.eps = eps
        self.n = n

    __call__ = zt.Estimator.joint

    def estimate(self, model, batch, opt):
        grads = [torch.zeros_like(weight.param, dtype=torch.float32) for weight in self.targets]
        for _ in range(self.n):
            seeds = [zt.seed() for _ in self.targets]
            plus = self.perturbed(model, batch, seeds, self.eps)
            minus = self.perturbed(model, batch, seeds, -self.eps)

            projected_grad = (plus - minus).sum() / (2 * self.eps * plus.numel() * self.n)
            for grad, s in zip(grads, seeds):
                grad += zt.noise(s) * projected_grad

        for weight, grad in zip(self.targets, grads):
            opt.accumulate(weight.param, grad)

    def perturbed(self, model, batch, seeds, scale):  # the model's losses with every weight moved by scale * z(seed)
        with zt.perturb({weight.write_weight: self.perturb(s, scale) for weight, s in zip(self.targets, seeds)}):
            return model(*batch)

    def perturb(self, seed, scale):  # a weight write's change: add scale * z(seed), undone after the pass
        def change(params):
            add_noise(params, seed, scale)
            return lambda: add_noise(params, seed, -scale)
        return change
