import itertools

import torch

class Estimator:
    # Estimates each of its targets on its own: a ZoWeight for a weight, or a ZoModule for a module whose weights are
    # estimated together. estimate(model, batch, opt, target) is written for one; estimator(model, batch, opt) runs it
    # for each target in turn, with that target's hooks registered, and without autograd, so weights can be changed in
    # place.
    def __init__(self, targets):
        self.targets = targets

    def __call__(self, model, batch, opt):
        with torch.no_grad():
            for target in self.targets:
                self.run(target.hooks, self.estimate, model, batch, opt, target)

    def joint(self, model, batch, opt):  # for __call__ = joint: estimate(model, batch, opt) once, for all targets together
        with torch.no_grad():
            self.run([hook for target in self.targets for hook in target.hooks], self.estimate, model, batch, opt)

    def run(self, hooks, f, *args):  # f(*args) with the hooks registered
        for hook in hooks:
            hook.register()
        f(*args)
        for hook in hooks:
            hook.remove()


# Seeds for zt.noise, which draws the same noise again from the same seed

_seeds = itertools.count(1)

def seed():  # a fresh seed for one draw
    return next(_seeds)

def manual_seed(start):
    global _seeds
    _seeds = itertools.count(start)
