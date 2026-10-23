import torch
from torch.nn import functional as F

import zotorch as zt
from optimizers import dust

# dust.py with dust/dust_optimized.py's memory savings that ZoTorch does not already make. ZoTorch already draws noise
# again from its seed instead of keeping it, keeps only the current target's values, and hands each gradient to the
# optimizer as soon as it is estimated; a zt.CacheHook on each block keeps only the blocks' outputs from the clean pass.
# This adds:
#
#   1. Every draw runs in its own pass, instead of nrep draws on nrep copies of the batch.
#   2. DustHead recomputes one slab of logits at a time from the head's input, and each draw's loss from the clean exp
#      sums of the other slabs, instead of keeping the logits.

class DustDirect(dust.DustDirect):
    together = False

class DustLocal(dust.DustLocal):
    together = False

class DustHead(zt.Estimator):
    def __init__(self, model, name, softcap, sigma=0.05, draws=1024, slab=256):
        super().__init__([zt.ZoWeight(model, name)])
        self.softcap = softcap
        self.sigma = sigma
        self.draws = draws
        self.slab = slab

    def estimate(self, model, batch, opt, weight):
        y = batch[1]
        with zt.capture():
            losses = model(*batch)
        clean_loss = losses / losses.numel()
        x = weight.input.value
        slabs = weight.param.shape[0] // self.slab
        draws = self.draws // slabs
        slab_of, index = y // self.slab, (y.clamp(min=0) % self.slab).unsqueeze(-1)  # the target's slab, its column

        def raw(block):  # one slab's logits before the soft cap
            return F.linear(x, weight.param[block * self.slab:(block + 1) * self.slab]).float()

        def cap(logits):
            return self.softcap * torch.tanh(logits / self.softcap)

        sums, clean_target = [], torch.zeros_like(clean_loss)
        for block in range(slabs):
            logits = cap(raw(block))
            sums.append(logits.exp().sum(-1))
            clean_target = torch.where(slab_of == block, logits.gather(-1, index).squeeze(-1), clean_target)
        total = sum(sums)

        grad = torch.zeros_like(weight.param, dtype=torch.float32)
        for block in range(slabs):
            clean, others = raw(block), total - sums[block]
            error = torch.zeros_like(clean)
            for _ in range(draws):
                s = zt.seed()
                logits = cap(clean + zt.noise(s) * self.sigma)
                target = torch.where(slab_of == block, logits.gather(-1, index).squeeze(-1), clean_target)
                loss = torch.where(y != -1, (others + logits.exp().sum(-1)).log() - target, 0) / y.numel()
                reward = clean_loss - loss
                if y.shape[0] > 1:  # centered across the batch, per token
                    reward = reward - reward.mean(0)
                error -= zt.noise(s) * reward / (self.sigma * draws)
            grad[block * self.slab:(block + 1) * self.slab] = error.flatten(0, 1).T @ x.flatten(0, 1).float()
        opt.accumulate(weight.param, grad)
