import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from model.gpt import GPT
from optimizers import noise, noise_optimized

# dust/dust_noise.py's training on one GPU: Boost -> Cut estimators, ANVIL II (batched polar updates) on the block
# matrices and Adam (betas 0.5, 0.98) elsewhere, at batch 4, a warmdown over the last 40% to 30% of each rate, and the
# LM head initialized at std 0.02. End merge: the last 32 training sequences feed no update. From 60% of the epoch,
# every 5 updates, nine exponential means of the weights are kept. After the last update each part moves toward one
# mean, in full inside its 64 top input-covariance directions (for a table, the 512 most frequent tokens' rows) and 0,
# 1/2 or all of the way outside, times 0.75, 1 or 1.25, keeping whatever lowers the held-out loss. Every training row
# but the held-out 32 is consumed once.
#
#     python demo/train_noise.py --tokens 1M --data <dir> --output runs/noise-1m
#     python demo/train_noise.py --estimator noise-optimized ...   with noise_optimized.py's memory savings

ROWS = {'1M': 61 * 8, '10M': 610 * 8}  # the rows dust.py trains on


def load_rows(path, count):  # a token matrix or dust/prepare_data.py's chunked format, without its padded rows
    rows = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(rows, dict):
        rows = torch.cat([chunk.view(rows['batch_size'], rows['sequence_size'])[:valid]
                          for chunk, valid in zip(rows['chunks'], rows['valid_counts'])])
    return rows[:count].long()


# ------------------------------------------------------------------------------------------------------ optimizer


MAPS = (
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
)


RUN = (1194, 240, 50, 514)


def newton_schulz(x):  # the record's six quintic maps on a stack (B, n, n) of square matrices, in bfloat16
    x = x.bfloat16()
    for a, b, c in MAPS:
        s = x @ x.transpose(1, 2)
        x = a * x + (b * s + c * s @ s) @ x
    return x


def gram_newton_schulz(x, restarts=(2, 4)):
    # The same maps on wide matrices (n < m), iterated on the n x n Gram matrix (Gram Newton-Schulz, Dao lab 2026).
    # R is rebuilt from Q X before the maps in `restarts` to bound the float16 error.
    x = x.half()
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
    R, Q = x @ x.transpose(1, 2), eye
    for t, (a, b, c) in enumerate(MAPS):
        if t in restarts:
            x = Q @ x
            R, Q = x @ x.transpose(1, 2), eye
        Z = b * R + c * R @ R
        Q = Q @ Z + a * Q
        RZ = R @ Z + a * R
        R = Z @ RZ + a * RZ
    return Q @ x


def polar_batch(g, margin=1.1):  # approximate orthogonalization of a stack (B, r, c) of same-shape matrices
    tall = g.size(1) > g.size(2)
    x = g.transpose(1, 2) if tall else g
    x = x / (margin * x.norm(dim=(1, 2), keepdim=True) + 1e-6)
    x = (newton_schulz(x) if x.size(1) == x.size(2) else gram_newton_schulz(x)).float()
    return x.transpose(1, 2) if tall else x


@torch.compile(dynamic=False)
def anvil_update(g, w, fast, slow, energy, beta, fast_beta, slow_beta, weight, lane_beta, lr, wd, lanes: int,
                 ratio: float, margin: float):
    # one ANVIL II step for a stack of same-shape matrices; updates fast, slow and energy in place
    fast.lerp_(g, 1 - fast_beta)
    slow.lerp_(g, 1 - slow_beta)
    update = polar_batch(g.lerp(weight * fast + (1 - weight) * slow, beta), margin)
    energy.lerp_(update.square().mean(lanes + 1, keepdim=True), 1 - lane_beta)
    gained = update * energy.clamp_min(1e-10).rsqrt()
    update = gained * (update.norm(dim=(1, 2), keepdim=True) / gained.norm(dim=(1, 2), keepdim=True).clamp_min(1e-10))
    decay = (slow * w >= 0) * w * (wd * lr)
    return w - lr * ratio * (update + decay)


class Anvil(torch.optim.Optimizer):
    # ANVIL II from modded-nanogpt PR #360, applied to all same-shape matrices at once
    def __init__(self, params, lr, weight_decay, steps, lane_beta=0.9, beta_min=0.85, beta_max=0.93,
                 slow_beta=0.98, fast_beta=0.85, fast_weight=0.4385, polar_margin=1.1):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.total, self.warmup, self.cooldown, self.engage = (round(k * steps / RUN[0]) for k in RUN)
        self.lane_beta, self.beta_min, self.beta_max = lane_beta, beta_min, beta_max
        self.slow_beta, self.fast_beta, self.fast_weight = slow_beta, fast_beta, fast_weight
        self.polar_margin, self.stacks = polar_margin, {}
        self.t = 0

    def rail_beta(self):  # the scheduled fast-rail and Nesterov beta
        lo, hi, cool = self.beta_min, self.beta_max, self.total - self.cooldown
        if self.t < self.warmup:
            return lo + (hi - lo) * self.t / self.warmup
        if self.t > cool:
            return hi - (hi - lo) * (self.t - cool) / self.cooldown
        return hi

    def stacked_state(self, shape, params):
        # The moving averages of one shape group as stacked tensors. Each parameter's state holds views into them, so
        # the Boost -> Cut handoff, which copies per-parameter state, carries them over.
        key = tuple(params)
        stack = self.stacks.get(key)
        if stack is not None and all(self.state[p].get('fast') is not None for p in params):
            return stack
        lanes = 1 if shape[0] >= shape[1] else 0
        weights = torch.stack([p.float() for p in params])

        def gathered(key, like):
            held = [self.state[p].get(key) for p in params]
            return torch.stack(held).clone() if all(h is not None for h in held) else torch.zeros_like(like)

        stack = dict(fast=gathered('fast', weights), slow=gathered('slow', weights),
                     energy=gathered('energy', weights.mean(lanes + 1, keepdim=True)))
        for i, p in enumerate(params):
            self.state[p].update(fast=stack['fast'][i], slow=stack['slow'][i], energy=stack['energy'][i])
        self.stacks[key] = stack
        return stack

    @torch.no_grad()
    def step(self):
        beta, engaged = self.rail_beta(), self.t >= self.engage
        fast_beta, weight = (self.fast_beta, self.fast_weight) if engaged else (beta, 1.0)
        device = self.param_groups[0]['params'][0].device
        value = lambda v: torch.tensor(float(v), device=device)
        for group in self.param_groups:
            lr, wd = group['lr'], group['weight_decay']
            by_shape = {}
            for p in group['params']:
                by_shape.setdefault(tuple(p.shape), []).append(p)
            for shape, params in by_shape.items():
                stack = self.stacked_state(shape, params)
                new = anvil_update(torch.stack([p.grad.float() for p in params]), torch.stack([p.float() for p in params]),
                                   stack['fast'], stack['slow'], stack['energy'], value(beta), value(fast_beta),
                                   value(self.slow_beta), value(weight), value(self.lane_beta), value(lr), value(wd),
                                   1 if shape[0] >= shape[1] else 0, max(1, shape[0] / shape[1]) ** 0.5, self.polar_margin)
                torch._foreach_copy_(params, list(new.to(params[0].dtype).unbind(0)))
        self.t += 1


class Optimizers:  # ANVIL and Adam behind ZoTorch's accumulate and apply: estimates are summed into each grad
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]

    def accumulate(self, param, grad):
        grad = grad.to(param.dtype)
        param.grad = grad if param.grad is None else param.grad + grad

    def apply(self):
        for o in self.optimizers:
            o.step()
            o.zero_grad(set_to_none=True)


def make_model(device, seed, settings):
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = GPT(settings.model)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        # Output projections: uniform with the declared std / sqrt(fan_in).
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = settings.projection_std * math.sqrt(3 / module.weight.shape[1])
                nn.init.uniform_(module.weight, -scale, scale)
        nn.init.normal_(model.lm_head.weight, std=settings.head_std)
    return model.requires_grad_(False)


def make_optimizer(model, settings, steps):  # ANVIL II on the block matrices, Adam elsewhere
    matrices, groups = [], {}
    for name, parameter in model.named_parameters():
        if name.startswith('transformer.h.') and parameter.ndim == 2 and 've_gate' not in name:
            matrices.append(parameter)
            continue
        scale = 300 if ('wte' in name or 'value_embeds' in name) else 30 if 'lambdas' in name else 1
        groups.setdefault(scale, []).append(parameter)
    adam = torch.optim.AdamW(
        [{'params': ps, 'lr': settings.lr * scale} for scale, ps in groups.items()],
        betas=settings.betas, eps=1e-8, weight_decay=0.0,
    )
    anvil = Anvil(matrices, settings.anvil_lr, settings.anvil_wd, steps,
                  polar_margin=settings.polar_margin, fast_weight=settings.fast_weight)
    return Optimizers(anvil, adam)


def set_rate(optimizer, step, steps, warmdown, final=0.0):
    # the linear warmdown, by one-based training step, down to `final` of each rate
    left = (steps - step + 1) / steps
    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
        group['lr'] = group['initial_lr'] * (final + (1 - final) * min(1.0, left / warmdown if warmdown else 1.0))


# ------------------------------------------------------------------------------------------------------- training


def stage_schedule(rows, settings):
    # (phase, phase step, first row, last row, phase updates): disjoint data slices, Boost then Cut
    b1, b2 = settings.boost_batch, settings.cut_batch
    n1, n2 = rows // b1, rows // b2
    alignment = b2 // b1
    boost = round(settings.switch * n1) // alignment * alignment
    batches = [('boost', i, (i - 1) * b1, i * b1, n1) for i in range(1, boost + 1)]
    batches += [('cut', i, (i - 1) * b2, i * b2, n2) for i in range(boost * b1 // b2 + 1, n2 + 1)]
    return batches


def handoff_optimizer(old, new, phase_step):  # keeps ANVIL's rails and Adam's moments; restarts ANVIL's clock
    for new_o, old_o in zip(new.optimizers, old.optimizers, strict=True):
        new_o.state.update(old_o.state)
    new.optimizers[0].t = phase_step - 1


@torch.no_grad()
def holdout_loss(model, rows):  # the mean next-token loss over rows
    device = next(model.parameters()).device
    total = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in rows.split(8):
        batch = batch.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous()).sum()
        total[0] += loss.double()
        total[1] += batch[:, 1:].numel()
    return (total[0] / total[1]).item()


def merge_part(name):
    if name.startswith('transformer.h.') and 've_gate' not in name:
        return '.'.join(name.split('.')[2:5])  # one matrix of block i (attn.c_q, ..., mlp.c_proj)
    return 've_gates' if 've_gate' in name else name


HOLDOUT = 32  # training sequences that feed no update and score the end merge
MERGE_FROM, MERGE_EVERY = 0.6, 5  # the exponential means follow the weights from 60% of the epoch, every 5 updates
MERGE_DECAYS = (0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02, 0.01)  # weight of the newest weights in each mean
MERGE_RHOS = (0.0, 0.5, 1.0)  # how far a candidate moves outside the high-curvature subspace
MERGE_K, MERGE_TOKENS = 64, 512  # input-covariance directions per linear layer; frequent-token rows per table
MERGE_COEFS = (0.75, 1.0, 1.25)  # multiples of each move in the joint search


@torch.no_grad()
def curvature_bases(model, rows, k=MERGE_K):
    # Per linear layer, the top-k eigenvectors of its input covariance E[x x^T], and the token order of `rows` by
    # frequency (for the embedding tables), from forward passes of the current weights on `rows`.
    device = next(model.parameters()).device
    layers = [(name + '.weight', mod) for name, mod in model.named_modules()
              if isinstance(mod, nn.Linear) and 've_gate' not in name]
    acc, hooks = {}, []
    for key, mod in layers:
        def hook(module, inputs, output, key=key):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
            acc[key] = acc.get(key, 0) + (x.T @ x).double()
        hooks.append(mod.register_forward_hook(hook))
    counts = torch.zeros(model.config.vocab_size, dtype=torch.float64, device=device)
    for batch in rows.split(4):
        batch = batch.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous())
        counts += torch.bincount(batch[:, :-1].reshape(-1).long(), minlength=len(counts)).double()
    for h in hooks:
        h.remove()
    bases = {key: torch.linalg.eigh(acc[key])[1].flip(1)[:, :k].float().contiguous() for key, _ in layers}
    return bases, torch.argsort(counts, descending=True)


def subspace_move(name, move, bases, order):
    # the part of a weight move inside its high-curvature subspace: the top input-covariance directions of a linear
    # layer, the rows of the most frequent tokens of an embedding table; other tensors whole
    if name in bases:
        return (move @ bases[name]) @ bases[name].T
    if name == 'transformer.wte.weight' or name.startswith('value_embeds.'):
        mask = torch.zeros(move.shape[0], 1, device=move.device, dtype=move.dtype)
        mask[order[:MERGE_TOKENS]] = 1
        return move * mask
    return move


@torch.no_grad()
def joint_parts(model, named, params, final, move, candidates, rows, plain, load, sweeps=2, shortlist=8):
    # The joint search over keys (rho, j, c) = final + c * move(rho, j) per part (None = the final weights): the best
    # single key for every part, then coordinate sweeps (all keys per part, then each part's best `shortlist`).
    parts = sorted({merge_part(n) for n, _ in named})
    keys = [None] + [(rho, j, c) for rho, j in candidates for c in MERGE_COEFS]
    cache = {}

    def value(i, key):
        if key is None:
            return final[i]
        if (i, key[0], key[1]) not in cache:
            cache[i, key[0], key[1]] = move(i, key[0], key[1])
        return final[i] + key[2] * cache[i, key[0], key[1]]

    def state(choice):
        return [value(i, choice[merge_part(n)]) for i, (n, _) in enumerate(named)]
    best_key, best = None, plain
    for key in keys[1:]:
        load([value(i, key) for i in range(len(params))])
        loss = holdout_loss(model, rows)
        if loss < best:
            best_key, best = key, loss
    choice = {p: best_key for p in parts}
    ranked = {}
    for sweep in range(sweeps):
        for p in parts:
            tried = []
            for key in (keys if sweep == 0 else ranked[p]):
                if key == choice[p]:
                    tried.append((best, key))
                    continue
                trial = dict(choice)
                trial[p] = key
                load(state(trial))
                loss = holdout_loss(model, rows)
                tried.append((loss, key))
                if loss < best:
                    best, choice = loss, trial
            if sweep == 0:
                ranked[p] = [k for _, k in sorted(tried, key=lambda x: x[0])[:shortlist]]
    load(state(choice))
    return dict(holdout_plain=plain, holdout_merged=best,
                merge_choices={p: (None if k is None else [k[0], MERGE_DECAYS[k[1]], k[2]]) for p, k in choice.items()})


@torch.no_grad()
def end_merge(model, means, rows):
    # Per part, candidates final + c * (inside + rho * (full - inside)), full = an exponential mean minus the final
    # weights and inside its part in the high-curvature subspace, chosen on the held-out loss by joint_parts.
    named = list(model.named_parameters())
    params = [q for _, q in named]
    final = [q.detach().float().clone() for q in params]
    bases, order = curvature_bases(model, rows)
    candidates = [(rho, j) for rho in MERGE_RHOS for j in range(len(means))]

    def move(i, rho, j):
        full = means[j][i] - final[i]
        if rho == 1.0:
            return full
        inside = subspace_move(named[i][0], full, bases, order)
        return inside + rho * (full - inside)

    def load(ws):
        for q, w in zip(params, ws):
            q.copy_(w.to(q.dtype))
    plain = holdout_loss(model, rows)
    return joint_parts(model, named, params, final, move, candidates, rows, plain, load)


def train(args, method, device):
    settings = method.Settings()
    training = load_rows(args.data / f'train_{args.tokens}.pt', ROWS[args.tokens])
    validation = load_rows(args.data / 'val.pt', 544)
    testing = load_rows(args.data / 'test.pt', 544)
    training = training[torch.randperm(len(training), generator=torch.Generator().manual_seed(args.seed + 1))]
    holdout_rows, training = training[-HOLDOUT:], training[:-HOLDOUT]  # these feed no update, only the end merge
    means = None  # the end merge's exponential means, one per MERGE_DECAYS entry
    schedule = stage_schedule(len(training), settings)
    steps = min(args.steps or len(schedule), len(schedule))
    args.output.mkdir(parents=True, exist_ok=True)

    def estimator_for(model, phase):
        return (method.Boost if phase == 'boost' else method.Cut)(model, settings, args.seed)

    def step(model, estimator, optimizer, batch):  # one update; returns the clean loss
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        estimator(model, (x, y), optimizer)
        optimizer.apply()
        return estimator.capture.clean_loss[y != -1].mean().item()

    # Compile every active phase on random tokens, then recreate all model, optimizer and RNG state.
    phases = {phase: horizon for phase, _, _, _, horizon in schedule}
    noise = torch.randint(0, settings.model.vocab_size,
                          (max(settings.boost_batch, settings.cut_batch), settings.model.sequence_len + 1),
                          generator=torch.Generator().manual_seed(args.seed))
    for phase, horizon in phases.items():
        model = make_model(device, args.seed, settings)
        size = settings.boost_batch if phase == 'boost' else settings.cut_batch
        step(model, estimator_for(model, phase), make_optimizer(model, settings, horizon), noise[:size].to(device))
        del model
    torch.cuda.empty_cache()

    phase = schedule[0][0]
    model = make_model(device, args.seed, settings)
    optimizer, estimator = make_optimizer(model, settings, phases[phase]), estimator_for(model, phase)
    log = (args.output / 'metrics.jsonl').open('a')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, paused = time.perf_counter(), 0.
    for n in range(1, steps + 1):
        current, phase_step, begin, end, horizon = schedule[n - 1]
        if current != phase:
            next_optimizer = make_optimizer(model, settings, horizon)
            handoff_optimizer(optimizer, next_optimizer, phase_step)
            optimizer, estimator, phase = next_optimizer, estimator_for(model, current), current
        set_rate(optimizer, phase_step, horizon, settings.warmdown, settings.final)
        loss = step(model, estimator, optimizer, training[begin:end].to(device))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start - paused
        record = dict(step=n, phase=phase, phase_step=phase_step, seqs=end,
                      tokens=end * settings.model.sequence_len, train_loss=loss, train_seconds=round(elapsed, 2),
                      peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))
        if args.eval_every and n % args.eval_every == 0 and n < steps:
            pause = time.perf_counter()
            record['val_loss'] = holdout_loss(model, validation)
            paused += time.perf_counter() - pause
        print(json.dumps(record), flush=True)
        log.write(json.dumps(record) + '\n')
        log.flush()
        if (n % MERGE_EVERY == 0 or n == steps) and begin / len(training) >= MERGE_FROM:
            with torch.no_grad():
                if means is None:
                    means = [[q.detach().float().clone() for q in model.parameters()] for _ in MERGE_DECAYS]
                else:
                    for a, mean in zip(MERGE_DECAYS, means):
                        for m, q in zip(mean, model.parameters()):
                            m.lerp_(q.detach().float(), a)
    log.close()
    extra = {}
    if means:  # the end merge, timed with the training (the plain weights' validation is not)
        pause = time.perf_counter()
        extra['val_loss_plain'] = holdout_loss(model, validation)
        paused += time.perf_counter() - pause
        merge_start = time.perf_counter()
        extra.update(end_merge(model, means, holdout_rows))
        torch.cuda.synchronize()
        extra['merge_seconds'] = round(time.perf_counter() - merge_start, 2)
        elapsed = time.perf_counter() - start - paused
    result = dict(val_loss=holdout_loss(model, validation), test_loss=holdout_loss(model, testing),
                  train_seconds=round(elapsed, 2), steps=steps, seqs=end, tokens=end * settings.model.sequence_len,
                  estimator=args.estimator, peak_cuda_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
                  **extra)
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, args.output / 'final.pt')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator', choices=['noise', 'noise-optimized'], default='noise')
    parser.add_argument('--tokens', choices=ROWS, default='1M')
    parser.add_argument('--data', type=Path, default=Path('/data'))
    parser.add_argument('--output', type=Path, default=Path('runs/noise'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--steps', type=int, help='a prefix of the epoch; the learning rate schedule is the full one')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    train(args, noise_optimized if args.estimator == 'noise-optimized' else noise, torch.device('cuda'))
