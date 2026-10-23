import torch

import zotorch as zt

# DUST - https://qlabs.sh/research/dust

def jitter(seeds, sigma):  # a write hook's change for the batch repeated once per seed: copy k adds sigma * z(seeds[k])
    def change(value):
        value = value.clone()
        for s, copy in zip(seeds, value.chunk(len(seeds))):
            copy += zt.noise(s) * sigma
        return value
    return change

class Dust(zt.Estimator):
    together = True  # a chunk's nrep draws run in one pass, on nrep copies of the batch

    def __init__(self, targets, sigma, nrep):
        super().__init__(targets)
        self.sigma = sigma
        self.nrep = nrep

    def loss(self, model, batch, copies=1):  # [copies, b, t]: per-token losses, each copy scaled to sum to its mean
        losses = model(*(torch.cat([x] * copies) for x in batch))
        losses = losses.view(copies, -1, *losses.shape[1:])
        return losses / losses[0].numel()

    def direct_errors(self, model, batch, clean_loss, writers, values, draws, sigma):
        # Each writer jitters its site, whose clean value is in values; every draw jitters them all. Rewards are
        # centered across each chunk of nrep draws.
        n = self.nrep
        copies = n if self.together else 1
        errors = [torch.zeros_like(value, dtype=torch.float32) for value in values]
        for _ in range(draws // n):
            seeds = [[zt.seed() for _ in writers] for _ in range(n)]
            rewards = []
            for start in range(0, n, copies):
                together = seeds[start:start + copies]
                with zt.perturb({writer: jitter([draw[i] for draw in together], sigma)
                                 for i, writer in enumerate(writers)}):
                    rewards += clean_loss - self.loss(model, batch, copies)
            if n > 1:
                mean = sum(rewards) / n
                rewards = [reward - mean for reward in rewards]
            for draw, reward in zip(seeds, rewards):
                for error, s in zip(errors, draw):
                    error -= zt.noise(s) * reward / (sigma * draws)
        return errors


class DustDirect(Dust):
    # direct_errors at the outputs of the named Linear or Embedding weights' layers, all jittered in each draw.
    def __init__(self, model, names, sigma=0.2, draws=8, nrep=2):
        super().__init__([zt.ZoWeight(model, name) for name in names], sigma, nrep)
        self.draws = draws

    __call__ = zt.Estimator.joint

    def estimate(self, model, batch, opt):
        with zt.capture():
            clean_loss = self.loss(model, batch)[0]
        errors = self.direct_errors(model=model,
                                    batch=batch,
                                    clean_loss=clean_loss,
                                    writers=[weight.write_output for weight in self.targets],
                                    values=[weight.output.value for weight in self.targets],
                                    draws=self.draws,
                                    sigma=self.sigma)
        for weight, error in zip(self.targets, errors):
            opt.accumulate(weight.param, zt.weight_gradient(weight.module, weight.input.value, error))

class DustLocal(Dust):
    def __init__(self, model, name, read, layers, arguments, draws, heads=1, sigma=0.05, target_draws=8,
                 target_sigma=0.2, nrep=2, gamma=0.98, window=None, uncredited=(), two_sided=()):
        super().__init__([zt.ZoModule(model, name, read, layers, arguments)], sigma, nrep)
        self.draws = draws
        self.heads = heads
        self.target_draws = target_draws
        self.target_sigma = target_sigma
        self.gamma = gamma
        self.window = window
        self.uncredited = uncredited
        self.two_sided = two_sided

    def estimate(self, model, batch, opt, module):
        # clean run
        with zt.capture():
            clean_loss = self.loss(model, batch)[0]

        o = module.read.input.value
        b, t, h = o.shape[0], o.shape[1], self.heads
        [g] = self.direct_errors(model, batch, clean_loss, [module.read.write_input], [o], self.target_draws,
                                 self.target_sigma)
        g = g.view(b, t, h, -1)  # dL/do, by head
        baseline = module.replay()
        credit = self.credit(t, baseline.device)

        for kind, weight in module.weights.items():
            draws = self.draws[kind]
            error = torch.zeros_like(weight.output.value, dtype=torch.float32).view(b, t, h, -1)
            for _ in range(draws):
                s = zt.seed()
                delta, divisor = self.delta(module, weight.write_output, kind in self.two_sided, s, baseline)
                score = (g * delta.view(b, t, h, -1)).sum(-1)  # [b, t, h]
                if kind not in self.uncredited:
                    score = torch.einsum('bsh,ts->bth', score, credit)
                error += zt.noise(s) * score / (divisor * draws)
            opt.accumulate(weight.param, zt.weight_gradient(weight.module, weight.input.value, error))

    def delta(self, module, writer, two_sided, s, baseline):  # the change at read's input from a jitter, its divisor
        with writer.perturb(jitter([s], self.sigma)):
            plus = module.replay()
        if not two_sided:   # dust
            return plus - baseline, self.sigma
        with writer.perturb(jitter([s], -self.sigma)):    # mezo
            return plus - module.replay(), 2 * self.sigma

    def credit(self, t, device):  # [t, s]: how much of output s's score goes to the jitter at t, in the causal window
        window = t if self.window is None else self.window
        positions = torch.arange(t, device=device)
        lag = positions[None, :] - positions[:, None]  # [perturbation position, output position]
        return ((lag >= 0) & (lag <= window)) * self.gamma ** lag.clamp(min=0)


class DustHead(Dust):
    # Each chunk of nrep draws jitters one slab of the clean logits, on nrep copies, and each draw's loss is
    # recomputed from the clean exp sums of the other slabs.
    def __init__(self, model, name, softcap, sigma=0.05, draws=1024, nrep=2, slab=256):
        super().__init__([zt.ZoWeight(model, name)], sigma, nrep)
        self.softcap = softcap
        self.draws = draws
        self.slab = slab

    def estimate(self, model, batch, opt, weight):
        y = batch[1]
        with zt.capture():
            clean_loss = self.loss(model, batch)[0]

        raw = weight.output.value
        b, t, vocab = raw.shape
        n, slabs = self.nrep, vocab // self.slab
        logits = self.cap(raw.float())
        sums = logits.exp().view(b, t, slabs, self.slab).sum(-1)
        total = sums.sum(-1)
        clean_target = logits.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        slab_of, index = y // self.slab, (y.clamp(min=0) % self.slab).unsqueeze(-1)  # the target's slab, its column

        error = torch.zeros_like(raw, dtype=torch.float32)
        for chunk in range(self.draws // n):
            block = chunk % slabs
            columns = slice(block * self.slab, (block + 1) * self.slab)
            seeds = [zt.seed() for _ in range(n)]
            changed = self.cap(jitter(seeds, self.sigma)(raw[..., columns].float().repeat(n, 1, 1)))
            partition = (total - sums[..., block]).repeat(n, 1) + changed.exp().sum(-1)
            target = torch.where(slab_of.repeat(n, 1) == block, changed.gather(-1, index.repeat(n, 1, 1)).squeeze(-1),
                                 clean_target.repeat(n, 1))
            loss = torch.where(y.repeat(n, 1) != -1, partition.log() - target, 0).view(n, b, t) / y.numel()
            rewards = clean_loss - loss
            if b > 1:  # centered across the batch, per token
                rewards = rewards - rewards.mean(1, keepdim=True)
            for s, reward in zip(seeds, rewards):
                error[..., columns] -= zt.noise(s) * reward / (self.sigma * (self.draws // slabs))
        opt.accumulate(weight.param, zt.weight_gradient(weight.module, weight.input.value, error))

    def cap(self, logits):
        return self.softcap * torch.tanh(logits / self.softcap)
