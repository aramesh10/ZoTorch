from dataclasses import dataclass, field
from functools import cache, partial
from math import gcd

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

import zotorch as zt
from model import gpt
from model.gpt import GPTConfig, apply_rotary_emb, has_ve, norm

# dust/dust_noise.py: Boost -> Cut, the val record of qlabs-eng/noise (train/val/run.py at b6405a1), as ZoTorch
# estimators on one GPU. The clean pass is read by ZoTorch readers and each estimate goes to opt.accumulate; the
# estimates are dust_noise.py's. Relative to dust.py:
#
#   - Orthogonal draws: every probe is a row of a sign-flipped Hadamard matrix, each token starting at its own row, so a
#     full cycle of draws is an exact orthogonal basis.
#   - Coordinate head: the head's error is each token's central loss difference along each of its own logits, in closed
#     form, at the final readout and (during Boost) at every intermediate readout.
#   - Boost (the first 20% of updates): stagewise credit. Each sub-block's output is credited against the shared head
#     reading the stream right after it, measured exactly along a full cycle (cycle_derivatives); the MLP hidden units
#     by their own central differences; the embedding through the first attention stage along its own path.
#   - Cut (the rest): forward loss probes. Every mid stream and the embedding is probed along a full cycle, carried to
#     the loss through every later block with K/V held, each MLP restricted to its token's top-512 units, and PR #19's
#     own-query attention response in every entered block, moving only the 256 strongest key logits per head. The
#     mixing scalars are probed along the same chains, and also take the input-path credit.
#   - Attention internals, one local step from the measured hub targets: queries by a full per-head cycle against each
#     query's own target; keys by single-key experiments, exact up to the step; values and VE gates in closed form.
#   - PR #23's value path and the independent input-K and VE-gate probes add input credit along linear consumers only;
#     corrected credits never become another probe's target.
#   - Future probes: 128 central pairs per source token for the first four blocks' hidden and Q outputs and the
#     embedding, scored at the original measured attention hubs of every later block.
#
# It keeps dust/dust_noise_optimized.py's memory savings, which leave the estimates unchanged: each query's routed keys
# are kept as indices into the clean keys, each MLP's routed units are gathered as the replay enters its layer,
# attention maps are built one sequence at a time and never cached, captured inputs keep their dtype, and the query
# probes are signed in place. noise_optimized.py's settings add its smaller replay blocks and embedding chunks.
#
# The optimizer, schedule and end merge are in demo/train_noise.py.

EPS = torch.finfo(torch.float32).eps
KEY_BUCKET = 256
HADAMARDS = {}
torch._dynamo.config.cache_size_limit = 64
flex = torch.compile(flex_attention)


@cache
def window_mask(window, length, device):
    return create_block_mask(
        lambda b, h, q, k: (q >= k) & (q - k <= window), None, None, length, length, device=device,
    )


def attention(q, k, v, window):  # model.attention with flex attention, which skips the blocks outside the window
    mask = window_mask(window, q.shape[1], q.device)
    return flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask).transpose(1, 2)


gpt.attention = attention  # the model's own forward uses it too, as in the record's model


@dataclass
class Settings:
    model: GPTConfig = field(default_factory=GPTConfig)
    switch: float = 0.2  # the share of updates in Boost
    boost_batch: int = 4
    cut_batch: int = 4
    projection_std: float = 1.5
    lr: float = 0.0021
    betas: tuple = (0.5, 0.98)
    warmdown: float = 0.4
    final: float = 0.3  # the warmdown ends at this fraction of each rate; the end merge anneals the rest
    head_std: float = 0.02  # the LM head's init std
    anvil_lr: float = 0.026
    anvil_wd: float = 2.25
    fast_weight: float = 1.0  # ANVIL fast rail
    polar_margin: float = 1.1
    sigma: float = 0.005
    fc_sigma: float = 0.05
    query_draws: int = 64
    query_sigma: float = 0.1
    nrep: int = 64  # query draws run together
    emb_chunk: int = 16  # embedding moves run together
    scalar_step: float = 0.01
    chunk: int = 2 ** 27
    chain_sigma: float = 0.01
    route_k: int = 512
    chain_tokens: int = 512  # tokens per replay block
    key_path_draws: int = 512  # independent input-K central pairs per token
    key_path_sigma: float = 0.01  # relative input RMS step
    key_path_tokens: int = 64  # source-token chunk; every allowed query contributes
    gate_path_sigma: float = 0.01  # relative input RMS step
    future_draws: int = 128  # independent hidden/Q/WTE probes scored at every later raw hub
    future_layers: int = 4  # first blocks whose hidden and Q outputs receive this correction
    future_tokens: int = 128
    future_chunk: int = 512
    future_sigma: float = 0.01  # source-relative RMS; Q includes its own complete MLP response
    query_keys: int = 256  # moving key logits per head
    key_sigma: float = 0.003


# ---------------------------------------------------------------------------------------------------------- helpers


@torch.compile
def coordinate_differences(raw, y, sigma):
    # For every token and every logit j: the loss with raw logit j moved up by sigma minus the loss with it moved down,
    # over 2 sigma. Moving one logit changes the partition function Z by one term, so the moved loss differs from the
    # clean loss by log1p(exp(z_j) / Z * expm1(z_j' - z_j)), and by -(z_j' - z_j) at the target logit. exp(z_j) / Z are
    # the clean forward's probabilities, used only to evaluate the perturbed losses. Zero at ignored tokens (y < 0).
    cap = lambda v: 15 * torch.tanh(v / 15)
    z, up, down = cap(raw.float()), cap(raw.float() + sigma), cap(raw.float() - sigma)
    weight = torch.softmax(z, -1)
    moved_loss = lambda new: torch.log1p(weight * torch.expm1(new - z))
    change = moved_loss(up) - moved_loss(down)
    target = y.clamp(min=0).unsqueeze(-1)
    change = change.scatter_add(-1, target, -(up - down).gather(-1, target))  # the target logit's own term
    return change / (2 * sigma) * (y >= 0).unsqueeze(-1)


def hadamard(d, device, dtype):  # the Sylvester Hadamard matrix of order d (a power of two), built once per process
    if (d, device, dtype) not in HADAMARDS:
        H = torch.ones(1, 1, device=device)
        while H.shape[0] < d:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        HADAMARDS[d, device, dtype] = H.to(dtype)  # +-1 is exact in bfloat16
    return HADAMARDS[d, device, dtype]


def chunk_size(draws, cap):  # the largest divisor of draws up to cap
    return next(n for n in range(min(cap, draws), 0, -1) if draws % n == 0)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return torch.cat((x[..., :half] * cos + x[..., half:] * sin,
                      -x[..., :half] * sin + x[..., half:] * cos), -1)


def norm_difference(x, delta):  # norm(x + delta) - norm(x), exactly: x (m, W) clean, delta (m, P, W) moves
    sq = x.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', x, delta)
    inv0 = torch.rsqrt(sq / x.shape[-1] + EPS)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / x.shape[-1] + EPS)
    return delta * inv.unsqueeze(-1) + x.unsqueeze(1) * (inv - inv0).unsqueeze(-1)


def norm_delta(x, delta):  # norm(x + delta) - norm(x), broadcasting x against delta
    width = x.shape[-1]
    eps = torch.finfo(x.dtype).eps
    square = x.square().sum(-1, keepdim=True)
    inverse = torch.rsqrt(square / width + eps)
    moved_inverse = torch.rsqrt(
        (square + 2 * (x * delta).sum(-1, keepdim=True)
         + delta.square().sum(-1, keepdim=True)) / width + eps)
    return delta * moved_inverse + x * (moved_inverse - inverse)


def decode_local_scores(scores, directions):
    # scores [..., P] times directions [..., P, D] (or shared [P, D]), summed over P, inside the caller's compiled
    # kernel so that no score-times-direction tensor is materialized
    if directions.ndim == 2:
        return scores @ directions
    return (scores.unsqueeze(-1) * directions).sum(-2)


@torch.compile(dynamic=False)
def decode_local_response(scores, steps, directions):  # central +/- scores, decoded along their directions
    probes = directions.shape[-2]
    directional = (scores[..., :probes] - scores[..., probes:]) / (2 * steps)
    return decode_local_scores(directional, directions)


# ----------------------------------------------------------------------------------------------- readouts (Boost)


@torch.compile(dynamic=False)
def cycle_derivatives(base, C, y, steps, sq, dots, dir_sq, width):
    # Forward differences of each token's readout loss along every direction D_j of a full cycle: base (n, V) W x of the
    # clean streams; C (P, V) row j is W D_j; steps (n, 1); sq (n, 1) |x|^2; dots (n, P) x . D_j; dir_sq (P,) |D_j|^2.
    # The moved stream's RMS comes from these, so the head's norm is evaluated exactly at every probe.
    target = y.clamp(min=0)
    base_y = base.gather(1, target.unsqueeze(1))
    dirs_y = C.T[target]
    inv0 = torch.rsqrt(sq / width + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(base_y * inv0 / 15)
    inv = torch.rsqrt((sq + 2 * steps * dots + steps * steps * dir_sq) / width + EPS)
    z = (base.unsqueeze(1) + steps.unsqueeze(-1) * C) * inv.unsqueeze(-1)
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh((base_y + steps * dirs_y) * inv / 15)
    return (moved - clean) / steps * (y >= 0).unsqueeze(1)


@torch.compile(dynamic=False)
def hidden_local_credit(u, steps, g, out_weight):
    # The full Hadamard central-difference cycle for the separable hidden objective, collapsed: H.T @ H / k is the
    # identity, so decoding leaves (g @ W_out) times each unit's scalar finite difference, ReLU crossings included.
    up, down = F.relu(u + steps), F.relu(u - steps)
    slope = (up - down) * (up + down) / (2 * steps)
    return (g @ out_weight) * slope


def frozen_attention(q, k_own, v_own, k_clean, v_clean, window):
    # The attention each token would see if only its own state moved: every query attends to the other positions' clean
    # keys and values and to its own moved ones, by swapping its own clean key and value through the log-sum-exp of one
    # attention over the clean keys and values. Tensors are [batch, token, head, dim].
    mask = window_mask(window, q.shape[1], q.device)
    out, lse = flex(q.transpose(1, 2), k_clean.transpose(1, 2), v_clean.transpose(1, 2), block_mask=mask,
                    return_lse=True)
    out, lse = out.transpose(1, 2).float(), lse.transpose(1, 2).float()
    scale = q.shape[-1] ** -0.5
    clean_self = torch.exp((q.float() * k_clean.float()).sum(-1) * scale - lse)
    own_self = torch.exp((q.float() * k_own.float()).sum(-1) * scale - lse)
    shift = own_self.unsqueeze(-1) * (v_own.float() - out) - clean_self.unsqueeze(-1) * (v_clean.float() - out)
    return (out + shift / (1 - clean_self + own_self).unsqueeze(-1)).to(q.dtype)

frozen_attention_compiled = torch.compile(frozen_attention)


@torch.compile(dynamic=False)
def embedding_moves(e, steps, rows, wq, wk, wv, cos, sin):  # the first stage's inputs with the embedding moved
    n, (b, t, width) = rows.shape[0], e.shape
    hd = 2 * cos.shape[-1]
    x0 = norm(e.unsqueeze(0) + steps.unsqueeze(0) * rows[:, None, None, :]).reshape(n * b, t, width)
    heads = lambda z: z.view(n * b, t, -1, hd)
    q = norm(apply_rotary_emb(heads(F.linear(x0, wq)), cos, sin))
    k = norm(apply_rotary_emb(heads(F.linear(x0, wk)), cos, sin))
    return x0, q, k, heads(F.linear(x0, wv))


@torch.compile(dynamic=False)
def readout_losses(x, weight, y):
    logits = F.linear(x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS), weight)
    logits = 15 * torch.tanh(logits / 15)
    return F.cross_entropy(logits, y.clamp(min=0), reduction='none') * (y >= 0)


@torch.compile(dynamic=False)
def stage_readout(x0, out, mix, wproj, weight, y):
    xmid = mix * x0 + F.linear(out.reshape(x0.shape).float(), wproj)
    return readout_losses(xmid.reshape(-1, xmid.shape[-1]), weight, y)


# -------------------------------------------------------------------------------------------- attention internals


@torch.compile(dynamic=False)
def moved_heads(raw, a, sigma, cos, sin):
    moved = (raw.unsqueeze(0) + sigma * a.float()).flatten(0, 1)
    return norm(apply_rotary_emb(moved, cos, sin)).to(a.dtype)


@torch.compile(dynamic=False)
def query_probe_heads(raw, a, sigma, cos, sin):  # the moved queries, with the probes grouped as heads
    moved = moved_heads(raw, a, sigma, cos, sin)
    batch, length, heads, width = raw.shape
    return moved.view(len(a), batch, length, heads, width).permute(1, 3, 0, 2, 4).reshape(
        batch, heads * len(a), length, width)


def query_probe_attention(moved, keys, values, window, count, batch):  # all probes share the clean K/V
    _, length, heads, width = keys.shape
    out = flex(moved, keys.transpose(1, 2), values.transpose(1, 2),
               block_mask=window_mask(window, length, moved.device), enable_gqa=True)
    return out.view(batch, heads, count, length, width).permute(2, 0, 3, 1, 4)


@torch.compile(dynamic=False)
def head_scores(out, baseline, hub):  # the moved-minus-clean outputs, scored against the hub
    change = out.float() - baseline.float()
    return (change * hub.float().unsqueeze(0)).sum(-1).permute(1, 3, 2, 0)


@torch.compile(dynamic=False)
def code_estimate(credited, a):
    return decode_local_scores(credited.permute(0, 2, 1, 3), a.float().permute(1, 2, 3, 0, 4))


@torch.compile(dynamic=False)
def raw_key_moves(kp, raw, steps, cos, sin, basis):  # each raw key moved along each basis direction, processed
    moved = raw.unsqueeze(1) + steps.unsqueeze(1) * basis[None, :, None, :]
    half = raw.shape[-1] // 2
    moved = torch.cat((moved[..., :half] * cos + moved[..., half:] * sin,
                       -moved[..., :half] * sin + moved[..., half:] * cos), -1)
    return moved * torch.rsqrt(moved.square().mean(-1, keepdim=True) + EPS) - kp.unsqueeze(1)


@torch.compile(dynamic=False)
def prepared_key_moves(x, steps, rows, projected_x, projected_rows, k_raw, cos, sin):
    # Central experiments reusing the linear projections of x and each direction, by the finite forward identity
    # N(x +/- s h) W - N(x) W = (inv_plusminus - inv0) (x W) +/- s inv_plusminus (h W), both RMSNorm responses included.
    # Positive and negative trials share one batched query/key product.
    m, probes, width = x.shape[0], rows.shape[0], x.shape[-1]
    heads, _, dh = k_raw.shape
    square = x.square().sum(-1, keepdim=True)
    dots = x @ rows.T
    moved_square = square + steps.square() * rows.square().sum(-1)
    inv0 = torch.rsqrt(square / width + EPS)
    inv_up = torch.rsqrt((moved_square + 2 * steps * dots) / width + EPS)
    inv_down = torch.rsqrt((moved_square - 2 * steps * dots) / width + EPS)
    up = ((inv_up - inv0).unsqueeze(-1) * projected_x.unsqueeze(1)
          + (steps * inv_up).unsqueeze(-1) * projected_rows.unsqueeze(0))
    down = ((inv_down - inv0).unsqueeze(-1) * projected_x.unsqueeze(1)
            - (steps * inv_down).unsqueeze(-1) * projected_rows.unsqueeze(0))
    raw_change = torch.cat((up, down), dim=1).reshape(m, 2 * probes, heads, dh).permute(2, 1, 0, 3)
    rotate = lambda z: torch.cat([z[..., :dh // 2] * cos + z[..., dh // 2:] * sin,
                                  -z[..., :dh // 2] * sin + z[..., dh // 2:] * cos], -1)
    r0 = rotate(k_raw)
    dr = rotate(raw_change)
    dk = norm_difference(r0.reshape(heads * m, dh),
                         dr.permute(0, 2, 1, 3).reshape(heads * m, 2 * probes, dh))
    return dk.reshape(heads, m, 2 * probes, dh).permute(0, 2, 1, 3)


@torch.compile(dynamic=False)
def decode_key_pairs(scores, steps, rows):
    return decode_local_response(scores.sum(0).T, steps, rows) / len(rows)


@torch.compile(dynamic=False)
def norm_move(x, v, sigma=0.01):
    # PR #23's central forward RMSNorm response along v: one nonlinear step, not an analytic VJP. The tiny floor defines
    # a finite response for a zero input or credit.
    s = sigma * (x.square().mean(-1, keepdim=True).clamp_min(1e-30)
                 / v.square().mean(-1, keepdim=True).clamp_min(1e-30)).sqrt()
    move = (s * v).unsqueeze(1)
    return (norm_difference(x, move)[:, 0] - norm_difference(x, -move)[:, 0]) / (2 * s)


@torch.compile(dynamic=False)
def prepared_gate_input_response(x, weight, kappa, rows, projected_rows, row_square, sigma):
    # token chunks share the direction projections and squared lengths
    steps = sigma * x.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    square = x.square().mean(-1, keepdim=True)
    cross = (x @ rows.T) / x.shape[-1]
    direction_square = row_square.unsqueeze(0)
    base = F.linear(x[:, :weight.shape[1]], weight).unsqueeze(1)
    direction = projected_rows.unsqueeze(0)
    positive = torch.rsqrt(square + 2 * steps * cross + steps.square() * direction_square + EPS)
    negative = torch.rsqrt(square - 2 * steps * cross + steps.square() * direction_square + EPS)
    zp = (base + steps.unsqueeze(-1) * direction) * positive.unsqueeze(-1)
    zm = (base - steps.unsqueeze(-1) * direction) * negative.unsqueeze(-1)
    response = ((torch.sigmoid(zp) - torch.sigmoid(zm)) * kappa.unsqueeze(1)).sum(-1) / steps
    return decode_local_scores(response, rows) / len(rows)


# ------------------------------------------------------------------------------------------------ replay (Cut)


@cache
def attention_kernels():  # finite-response reductions; neither kernel differentiates an attention operation
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def routed(Q, K, V, I, P, Rest, Out, Length,
               Probes: tl.constexpr, Heads: tl.constexpr, Keys: tl.constexpr, D: tl.constexpr,
               BP: tl.constexpr, BK: tl.constexpr):
        # each query's moved output from its selected keys and values, read through their indices
        token, head = tl.program_id(1) // Heads, tl.program_id(1) % Heads
        probes = tl.program_id(0) * BP + tl.arange(0, BP)
        keys, features = tl.arange(0, BK), tl.arange(0, D)
        q = tl.load(Q + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                    probes[:, None] < Probes, 0)
        index = tl.load(I + (token * Heads + head) * Keys + keys, keys < Keys, 0)
        offsets = (head * Length + index[:, None]) * D + features[None, :]
        k = tl.load(K + offsets, keys[:, None] < Keys, 0)
        v = tl.load(V + offsets, keys[:, None] < Keys, 0)
        p = tl.load(P + (token * Heads + head) * Keys + keys, keys < Keys, 0)
        # Match both BF16 matmul output roundings of the original implementation.
        ds = tl.dot(q, tl.trans(k)).to(q.dtype).to(tl.float32) * (D ** -.5)
        dlse = libdevice.log1p(tl.sum(p[None, :] * libdevice.expm1(ds), 1))
        weights = (p[None, :] * libdevice.expm1(ds - dlse[:, None])).to(v.dtype)
        moved = tl.dot(weights, v).to(v.dtype).to(tl.float32)
        rest = tl.load(Rest + (token * Heads + head) * D + features)
        moved += libdevice.expm1(-dlse)[:, None] * rest[None, :]
        tl.store(Out + ((token * Probes + probes[:, None]) * Heads + head) * D + features[None, :],
                 moved, probes[:, None] < Probes)

    @triton.jit
    def key_response(Q, DK, A, C, Out,
                     H: tl.constexpr, T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
                     QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
                     AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
                     CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
                     SourceStart, QueryStart, Window: tl.constexpr,
                     BC: tl.constexpr, BQ: tl.constexpr):
        # single-key experiments: o'_s - o_s = (a' - a)(v_t - o_s) / (1 - a + a'), a' = a exp(dl), scored by C
        head = tl.program_id(1)
        col = tl.program_id(0) * BC + tl.arange(0, BC)
        source = col % M
        features = tl.arange(0, D)
        changed = tl.load(DK + (head * P * M + col[None, :]) * D + features[:, None],
                          col[None, :] < P * M, 0)
        first = tl.maximum(0, SourceStart + tl.min(source, 0) - QueryStart)
        last = tl.minimum(T, SourceStart + tl.max(source, 0) + Window + 1 - QueryStart)
        first = first // BQ * BQ
        result = tl.full((BC,), 0, tl.float32)
        for start in range(first, last, BQ):
            query = start + tl.arange(0, BQ)
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD,
                        query[:, None] < T, 0)
            delta = tl.dot(q, changed, input_precision='tf32x3') * (D ** -.5)
            valid = (query[:, None] < T) & (col[None, :] < P * M)
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, valid, 0)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, valid, 0)
            change = a * libdevice.expm1(delta)
            result += tl.sum(change / (1 + change) * c, 0)
        tl.store(Out + head * P * M + col, result, col < P * M)

    return routed, key_response


@torch.library.custom_op('zotorch_noise::routed_response', mutates_args=())
def routed_response(dq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, indices: torch.Tensor,
                    probabilities: torch.Tensor, rest: torch.Tensor) -> torch.Tensor:
    import triton
    m, probes, heads, width = dq.shape
    count = indices.shape[2]
    out = torch.empty((m, probes, heads, width), device=dq.device, dtype=torch.float32)
    attention_kernels()[0][(triton.cdiv(probes, 32), m * heads)](
        dq.contiguous(), keys, values, indices.contiguous(), probabilities.contiguous(), rest.contiguous(), out,
        keys.shape[1], probes, heads, count, width, 32, max(16, triton.next_power_of_2(count)), num_warps=4)
    return out


@routed_response.register_fake
def _(dq, keys, values, indices, probabilities, rest):
    return torch.empty(dq.shape, device=dq.device, dtype=torch.float32)


@torch.library.custom_op('zotorch_noise::key_response', mutates_args=())
def key_response(q: torch.Tensor, dk: torch.Tensor, A: torch.Tensor, C: torch.Tensor,
                 source_start: int, query_start: int, window: int) -> torch.Tensor:
    import triton
    heads, probes, sources, width = dk.shape
    out = torch.empty((heads, probes, sources), device=q.device, dtype=torch.float32)
    attention_kernels()[1][(triton.cdiv(probes * sources, 64), heads)](
        q, dk.contiguous(), A, C, out, heads, q.shape[1], sources, probes, width,
        *q.stride(), *A.stride(), *C.stride(), source_start, query_start, window,
        64, 64, num_warps=4, enable_fp_fusion=False)
    return out


@key_response.register_fake
def _(q, dk, A, C, source_start, query_start, window):
    return torch.empty(dk.shape[:3], device=q.device, dtype=torch.float32)


def query_change(x, delta, pos, w_q, keys, values, indices, top_p, rest, w_proj_t, cos, sin):
    # PR #19's finite own-query response: finite changes carried through both RMSNorms and the softmax, moving only the
    # selected key logits (read through their indices) and keeping the exact rescaling of the held output
    m, probes, _ = delta.shape
    heads, _, width = keys.shape
    c, s = cos[pos][:, None, None], sin[pos][:, None, None]
    rotate = lambda u: torch.cat([u[..., :width // 2] * c + u[..., width // 2:] * s,
                                  u[..., :width // 2] * -s + u[..., width // 2:] * c], -1)
    r0 = rotate((norm(x) @ w_q.T).view(m, 1, heads, width))
    dr = rotate((norm_difference(x, delta) @ w_q.T).view(m, probes, heads, width))
    sq = r0.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + EPS)
    inv = torch.rsqrt((sq + 2 * (r0 * dr).sum(-1, keepdim=True) + dr.square().sum(-1, keepdim=True)) / width + EPS)
    dq = dr * inv + r0 * (inv - inv0)
    dout = routed_response(dq.to(keys.dtype), keys, values, indices, top_p, rest)
    return dout.flatten(-2) @ w_proj_t


@torch.compile(dynamic=False)
def chain_chunk(site, move, scale, e, x0, mids, xL, idx, hs, lam_r, lam_0, w_fc, w_out_t, weight_t, base,
                target_logit_w, y, own):
    # Forward differences of each token's loss along every move of one site, K/V held, MLPs restricted to each token's
    # top-k units (gathered as each layer is entered). site: -1 for the raw embedding, else the block whose mid stream
    # moves. move (m, P, W); scale (m, P) the step each difference is divided by; e, x0, xL (m, W); mids per block
    # (m, W); idx, hs per block (m, k) unit indices and clean pre-activations; w_fc per block (H, W); w_out_t per block
    # (H, W); weight_t (W, V); base (m, V) clean top-stream logits before the norm; target_logit_w (m, W); own the
    # routed keys of every entered attention. Returns (m, P).
    pos, inputs, w_q, keys, values, indices, top_p, rest, w_proj_t, (cos, sin) = own
    if site < 0:
        dx0 = norm(e.unsqueeze(1) + move) - x0.unsqueeze(1)
        delta, first = (lam_r[0] + lam_0[0]) * dx0, 0
    else:
        dx0, delta, first = None, move, site
    for b in range(first, len(mids)):
        if b > first:
            delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
        if site < 0 or b > first:
            delta = delta + query_change(inputs[b], delta, pos, w_q[b], keys[b], values[b], indices[b],
                                         top_p[b], rest[b], w_proj_t[b], cos, sin)
        dz = norm_difference(mids[b], delta)
        fc, wo = w_fc[b][idx[b]], w_out_t[b][idx[b]]
        dh = torch.bmm(dz.to(fc.dtype), fc.transpose(1, 2)).float()
        h = hs[b].unsqueeze(1)
        up, down = F.relu(h + dh), F.relu(h)
        delta = delta + torch.bmm(((up - down) * (up + down)).to(wo.dtype), wo).float()
    sq = xL.square().sum(-1, keepdim=True)
    dots = torch.einsum('md,mpd->mp', xL, delta)
    inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / xL.shape[-1] + EPS)
    z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
    zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
    moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
    inv0 = torch.rsqrt(sq / xL.shape[-1] + EPS)
    clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
        base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
    return (moved - clean) / scale * (y >= 0).unsqueeze(1)


# ---------------------------------------------------------------------------------------------------- estimators


class Inputs(dict):  # layer inputs as captured, widened to float32 when read
    def __getitem__(self, name):
        value = super().__getitem__(name)
        return value.float() if value.is_floating_point() else value


class Capture:
    # The clean pass's values, from ZoTorch readers on every Linear's and Embedding's input and output and on every
    # block's input, and the record's orthogonal draws and coordinate head. hooks are the readers, for the estimator.
    def __init__(self, model, seed):
        self.model = model
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16
        self.generator = torch.Generator(self.device).manual_seed(seed + 3)
        self.shared_generator = torch.Generator(self.device).manual_seed(seed + 5)
        self.frames = {}
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))}
        self.readers = {n: (zt.ReaderHook(zt.Input(m, 0)), zt.ReaderHook(zt.Output(m)))
                        for n, m in self.modules.items()}
        self.block_readers = [zt.ReaderHook(zt.Input(block, 0)) for block in model.transformer.h]
        self.hooks = [*(reader for pair in self.readers.values() for reader in pair), *self.block_readers]

    def capture(self, x, y):
        self.x, self.y = x, y
        self.batch, self.length = x.shape
        self.frames.clear()
        with torch.autocast('cuda', dtype=self.dtype):
            with zt.capture():
                self.clean_loss = self.model(x, y).float()
            self.inputs = Inputs((n, i.value) for n, (i, _) in self.readers.items())
            self.outputs = {n: o.value for n, (_, o) in self.readers.items()}
            self.blocks = [reader.value for reader in self.block_readers]
            self.x0 = norm(self.outputs['transformer.wte'])

    def signs(self, *shape, shared=False):
        generator = self.shared_generator if shared else self.generator
        return torch.randint(0, 2, shape, device=self.device, generator=generator).to(self.dtype) * 2 - 1

    def orthogonal_noise(self, site, draws, chunk, n, b, t, d):
        # draws chunk * n .. chunk * n + n of this step's draws at `site`: rows of a sign-flipped Hadamard matrix,
        # orthogonal within each cycle
        if site not in self.frames:
            cycles = -(-draws // d)
            self.frames[site] = (self.signs(cycles, d, shared=True),
                                 torch.randint(0, d, (b, t), device=self.device, generator=self.shared_generator))
        patterns, offset = self.frames[site]
        index = chunk * n + torch.arange(n, device=self.device)
        rows = (offset.unsqueeze(0) + index[:, None, None]) % d
        a = hadamard(d, self.device, self.dtype)[rows]
        return a.mul_(patterns[index // d][:, None, None, :]).mul_(self.signs(n, b, t, 1))

    def head_error(self, sigma=0.1):
        raw = self.outputs['lm_head'][..., :self.model.config.vocab_size]
        error = coordinate_differences(raw, self.y, sigma)
        return F.pad(error, (0, self.outputs['lm_head'].shape[-1] - raw.shape[-1]))

    def reassemble(self, site, error):  # the weight gradient from the site's output error and its clean input
        return zt.weight_gradient(self.modules[site], self.inputs[site], error)


class Boost(zt.Estimator):
    # Phase 1: stagewise credit, each sub-block against the shared head reading the stream right after it. Its one
    # target is the Capture, so calling the estimator registers the readers around estimate.
    def __init__(self, model, settings, seed):
        self.capture = Capture(model, seed)
        super().__init__([self.capture])
        self.model, self.s = model, settings
        self.device = self.capture.device
        self.width = model.config.n_embd
        self.generator = torch.Generator(self.device).manual_seed(seed + 17)

    def estimate(self, model, batch, opt, capture):  # every weight's gradient of the mean loss, to opt
        x, y = batch
        gradients = self.gradients(x, y)
        valid = (y != -1).sum()
        for name, param in model.named_parameters():
            opt.accumulate(param, (gradients[name] / valid).float())

    def signs(self, d):
        return torch.randint(0, 2, (d,), device=self.device, generator=self.generator).float() * 2 - 1

    def basis(self):
        return hadamard(self.width, self.device, torch.float32) * self.signs(self.width)

    def probe(self, x, y):  # each token's readout credit, measured along a full cycle of the step's basis
        weight = self.model.lm_head.weight.float()[:self.model.config.vocab_size]
        H = self.basis()
        stream_dirs = H.T
        C = (weight @ stream_dirs).T.contiguous()
        dir_sq = stream_dirs.square().sum(0)
        steps = self.s.sigma * x.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        n, vocab = x.shape[0], C.shape[1]
        # each token's first row, used only by partial cycles, drawn either way so the random stream is the same
        torch.randint(0, self.width, (n, 1), device=self.device, generator=self.generator)
        credit = torch.zeros(n, self.width, device=self.device)
        per = max(1, self.s.chunk // (self.width * vocab))
        for t0 in range(0, n, per):
            xs, ys = x[t0:t0 + per], y[t0:t0 + per]
            derivative = cycle_derivatives(xs @ weight.T, C, ys, steps[t0:t0 + per],
                                           xs.square().sum(-1, keepdim=True), xs @ stream_dirs, dir_sq, self.width)
            credit[t0:t0 + per] = derivative @ H / self.width
        return credit

    def hidden_credit(self, u, g, out_weight):  # all hidden units' local central differences
        self.signs(256)  # the RNG advance of the retired hidden cycle
        steps = self.s.fc_sigma * u.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return hidden_local_credit(u, steps, g, out_weight.float())

    def embedding_block0(self, e, y, x):
        # the embedding's credit for the first attention stage's readout along each token's own path (every query sees
        # the other positions' clean keys and values and its own moved ones)
        model = self.model
        cfg = model.config
        attn = model.transformer.h[0].attn
        b, t = x.shape
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        window = model.window_sizes[0][0]
        mix = (model.resid_lambdas[0] + model.x0_lambdas[0]).float()
        weight = model.lm_head.weight.float()[:cfg.vocab_size]
        W = {name: getattr(attn, 'c_' + name).weight.float() for name in ('q', 'k', 'v', 'proj')}
        e = e.float().view(b, t, self.width)
        x0 = norm(e)
        k_clean = norm(apply_rotary_emb(F.linear(x0, W['k']).view(b, t, h, hd), cos, sin))
        v_clean = F.linear(x0, W['v']).view(b, t, h, hd)
        H = self.basis()
        steps = self.s.sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        derivative = torch.zeros(b, t, self.width, device=self.device)
        n = chunk_size(self.width, self.s.emb_chunk)
        yf = y.reshape(1, -1).expand(n, -1).reshape(-1)
        q0 = norm(apply_rotary_emb(F.linear(x0, W['q']).view(b, t, h, hd), cos, sin))
        out0 = frozen_attention(q0, k_clean, v_clean, k_clean, v_clean, window)
        xmid0 = mix * x0 + F.linear(out0.reshape(b, t, self.width).float(), W['proj'])
        clean = readout_losses(xmid0.reshape(-1, self.width), weight, y.reshape(-1)).view(1, b, t)
        for j0 in range(0, self.width, n):
            rows = H[j0:j0 + n]
            xm0, q, k, v = embedding_moves(e, steps, rows, W['q'], W['k'], W['v'], cos, sin)
            rep = lambda z: z.unsqueeze(0).expand(n, -1, -1, -1, -1).reshape(n * b, t, h, hd)
            out = frozen_attention_compiled(q, k, v, rep(k_clean), rep(v_clean), window)
            moved = stage_readout(xm0, out, mix, W['proj'], weight, yf).view(n, b, t)
            derivative[..., j0:j0 + n] = ((moved - clean) / steps.squeeze(-1)).permute(1, 2, 0)
        return (derivative @ H / self.width).reshape(b * t, self.width)

    def stage_losses(self, i, lam_r, lam_0, x_prev, x0, ve, y):  # the readout losses right after block i's attention
        model, block = self.model, self.model.transformer.h[i]
        T = x_prev.shape[1]
        cos_sin = (model.cos[:, :T], model.sin[:, :T])
        xin = lam_r * x_prev.float() + lam_0 * x0.float()
        xmid = xin + block.attn(norm(xin), None if ve is None else ve.float(), cos_sin, model.window_sizes[i]).float()
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        return readout_losses(xmid.float().flatten(0, 1), weight, y.flatten())

    def query_errors(self, draws, targets):  # a complete per-head query cycle, against each query's target
        d, model = self.capture, self.model
        cfg = model.config
        L, b, t = cfg.n_layer, d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        sigma = self.s.query_sigma
        n = chunk_size(draws, self.s.nrep)
        processed = lambda raw: norm(apply_rotary_emb(raw, cos, sin))
        dt = d.dtype
        errors = {}
        for i in range(L):
            prefix = f'transformer.h.{i}.attn.'
            site = prefix + 'c_q'
            window = model.window_sizes[i][0]
            q_raw, k_raw, v_raw = (d.outputs[prefix + 'c_' + name].float().view(b, t, h, hd) for name in 'qkv')
            ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd) if has_ve(i, L) else None
            z = d.outputs[prefix + 've_gate'].float() if has_ve(i, L) else None
            values = v_raw if ve is None else v_raw + (2 * torch.sigmoid(z)).unsqueeze(-1) * ve
            kp = processed(k_raw)
            target = targets[i].view(b, t, h, hd)
            baseline = attention(processed(q_raw).to(dt), kp.to(dt), values.to(dt), window)
            fixed, vals = kp.to(dt), values.to(dt)
            estimate = torch.zeros(b, t, h, hd, device=self.device)
            for c in range(draws // n):
                a = d.orthogonal_noise(site, draws, c, n, b, t, cfg.n_embd).view(n, b, t, h, hd)
                out = query_probe_attention(query_probe_heads(q_raw, a, sigma, cos, sin), fixed, vals, window, n, b)
                estimate += code_estimate(head_scores(out, baseline, target), a)
            errors[site] = (estimate / (draws * sigma)).reshape(b, t, cfg.n_embd)
        return errors

    def key_context(self, i, sequence, target):  # the clean FP32 single-key score context, shared by raw-K and input-K
        d, model = self.capture, self.model
        h, hd, t = model.config.n_head, self.width // model.config.n_head, d.length
        cos, sin = model.cos[0, :t, 0].float(), model.sin[0, :t, 0].float()
        rot = lambda z: torch.cat([z[..., :hd // 2] * cos + z[..., hd // 2:] * sin,
                                   -z[..., :hd // 2] * sin + z[..., hd // 2:] * cos], -1)
        unit = lambda z: z * torch.rsqrt(z.square().mean(-1, keepdim=True) + EPS)
        pos = torch.arange(t, device=self.device)
        lag = pos[:, None] - pos[None, :]
        allowed = ((lag >= 0) & (lag <= model.window_sizes[i][0])).float()
        prefix = f'transformer.h.{i}.attn.'
        heads = lambda name: d.outputs[prefix + name][sequence].float().view(t, h, hd).transpose(0, 1)
        q_raw, k_raw, values = heads('c_q'), heads('c_k'), heads('c_v')
        if has_ve(i, model.config.n_layer):
            ve = d.outputs[f'value_embeds.{i}'][sequence].float().view(t, h, hd).transpose(0, 1)
            gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][sequence].float().view(t, h)).T.unsqueeze(-1)
            values = values + gate * ve
        qp, kp = unit(rot(q_raw)), unit(rot(k_raw))
        logits = (qp @ kp.transpose(1, 2)) / hd ** .5
        A = torch.softmax(logits.masked_fill(allowed == 0, float('-inf')), -1)
        hub = target[sequence].view(t, h, hd).transpose(0, 1)
        C = (hub @ values.transpose(1, 2) - (hub * (A @ values)).sum(-1, keepdim=True)) * allowed
        return qp, kp, k_raw, cos, sin, A, C

    def key_errors(self, layer, context):
        # key credit, one local step from the target, by single-key experiments along every raw-K coordinate: exact up
        # to the one-sided step, with no interference between positions; masked query blocks are skipped
        qp, kp, raw, cos, sin, A, C = context
        h, t, hd = raw.shape
        result = torch.zeros_like(raw)
        eye = torch.eye(hd, device=self.device)
        steps = self.s.key_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        window = self.model.window_sizes[layer][0]
        for t0 in range(0, t, KEY_BUCKET):
            t1 = min(t0 + KEY_BUCKET, t)
            source, query = slice(t0, t1), slice(t0, min(t, t1 + window))
            moves = raw_key_moves(kp[:, source], raw[:, source], steps[:, source], cos[source], sin[source], eye)
            scored = key_response(qp[:, query], moves, A[:, query, source], C[:, query, source], t0, t0, window)
            result[:, source] = scored.transpose(1, 2) / steps[:, source]
        return result

    def sequence_map(self, i, r):  # the clean attention of block i for sequence r, A[1, h, query, source]
        d, model = self.capture, self.model
        cfg = model.config
        t, h, hd = d.length, cfg.n_head, cfg.n_embd // cfg.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        prefix = f'transformer.h.{i}.attn.'
        q, k = (norm(apply_rotary_emb(d.outputs[prefix + name][r:r + 1].float().view(1, t, h, hd), cos, sin))
                for name in ('c_q', 'c_k'))
        logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / hd ** .5
        positions = torch.arange(t, device=self.device)
        lag = positions[:, None] - positions[None, :]
        logits = logits.masked_fill(~((lag >= 0) & (lag <= model.window_sizes[i][0])), float('-inf'))
        return logits.softmax(-1).to(torch.bfloat16)

    def value_credit(self, i, target):  # fixed-attention value credit; all query tokens contribute, undiscounted
        d, cfg = self.capture, self.model.config
        b, t, h = d.batch, d.length, cfg.n_head
        hub = target.reshape(b, t, h, cfg.n_embd // h).transpose(1, 2).float()
        return torch.cat([self.sequence_map(i, r).float().transpose(-1, -2) @ hub[r:r + 1]
                          for r in range(b)]).transpose(1, 2)

    def internals(self, gradients, targets, key_errors, credits, value_credits):
        # The attention internals, one local step from the measured hub targets: queries by a query cycle against them,
        # keys from key_errors. A value move at t reaches query s's output as A[s, t] times the move, so the value
        # credit is sum_s A[s, t] target_s: value_credits, from corrected targets where there are any, as only V/VE may
        # use them; a gate move scales the value embedding by the gate's slope (the record's central difference, odd in
        # the move), against credits from the measured targets.
        d, cfg = self.capture, self.model.config
        b, t = d.batch, d.length
        h, hd = cfg.n_head, cfg.n_embd // cfg.n_head
        errors = self.query_errors(self.s.query_draws, targets)
        errors.update(key_errors)
        for i in targets:
            prefix = f'transformer.h.{i}.attn.'
            errors[prefix + 'c_v'] = value_credits[i].reshape(b, t, cfg.n_embd)
            if has_ve(i, cfg.n_layer):
                z = d.outputs[prefix + 've_gate'].float()
                ve = d.outputs[f'value_embeds.{i}'].float().view(b, t, h, hd)
                slope = (torch.sigmoid(z + 0.1) - torch.sigmoid(z - 0.1)) / 0.1
                errors[prefix + 've_gate'] = slope * (ve * credits[i]).sum(-1)
        for site, error in errors.items():
            gradients[site + '.weight'] = d.reassemble(site, error)
            if site.endswith('attn.c_v'):
                i = int(site.split('.')[2])
                if has_ve(i, cfg.n_layer):
                    gate = 2 * torch.sigmoid(d.outputs[f'transformer.h.{i}.attn.ve_gate'].float())
                    ve_error = (error.view(b, t, gate.shape[-1], -1) * gate.unsqueeze(-1)).view(b, t, -1)
                    gradients[f'value_embeds.{i}.weight'] = d.reassemble(f'value_embeds.{i}', ve_error)

    def gradients(self, x, y):  # {name: gradient of the summed token loss}
        d, model, s = self.capture, self.model, self.s
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {'lm_head.weight': d.reassemble('lm_head', d.head_error())}
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        x0 = d.x0
        targets, streams = {}, []
        for i in range(L):
            p = f'transformer.h.{i}.'
            xin = d.blocks[i]
            xmid = xin + d.outputs[p + 'attn.c_proj']
            xout = xmid + d.outputs[p + 'mlp.c_proj']
            streams.append((xin, xmid, xout))
            xm, xo = xmid.float().flatten(0, 1), xout.float().flatten(0, 1)
            block = model.transformer.h[i]
            g_mid = self.probe(xm, yf)
            gradients[p + 'attn.c_proj.weight'] = g_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            # One local linear step from the measured stream credit (PR #15).
            targets[i] = (g_mid @ block.attn.c_proj.weight.float()).view(b, t, self.width)
            # Keep later probes on the same RNG stream as the retired hub cycle.
            self.signs(self.width)
            torch.randint(0, self.width, (xm.shape[0], 1), device=self.device, generator=self.generator)
            g_out = self.probe(xo, yf)
            gradients[p + 'mlp.c_proj.weight'] = g_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, g_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            if i < L - 1:  # the head learns at every intermediate readout too
                for stream in (xm, xo):
                    normed = stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + EPS)
                    error = coordinate_differences(normed @ weight.T, yf, 0.1)
                    gradients['lm_head.weight'][:weight.shape[0]] += error.T @ normed
        key_errors = {f'transformer.h.{i}.attn.c_k': self.layer_key_errors(i, target) for i, target in targets.items()}
        credits = {i: self.value_credit(i, target) for i, target in targets.items()}
        self.internals(gradients, targets, key_errors, credits, credits)
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        emb = self.embedding_block0(e, yf.view(b, t), x)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', emb.view(b, t, self.width))
        scalars = torch.zeros(2, L, device=self.device)
        step = s.scalar_step
        for i in range(L):
            x_prev = x0 if i == 0 else streams[i - 1][2]
            ve = model.value_embeds[str(i)](x) if str(i) in model.value_embeds else None
            lr_, l0 = model.resid_lambdas[i], model.x0_lambdas[i]
            for which in range(2):
                up = self.stage_losses(i, lr_ + step * (which == 0), l0 + step * (which == 1), x_prev, x0, ve, y)
                down = self.stage_losses(i, lr_ - step * (which == 0), l0 - step * (which == 1), x_prev, x0, ve, y)
                scalars[which, i] = (up - down).sum() / (2 * step)
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        return gradients

    def layer_key_errors(self, i, target):  # block i's raw-K error, sequence by sequence
        d, h, hd = self.capture, self.model.config.n_head, self.width // self.model.config.n_head
        credit = torch.zeros(d.batch, h, d.length, hd, device=self.device)
        for r in range(d.batch):
            credit[r] = self.key_errors(i, self.key_context(i, r, target))
        return credit.permute(0, 2, 1, 3).reshape(d.batch, d.length, h * hd)


class Cut(Boost):
    # Phase 2: own-query forward loss probes, with PR #23's value path and the input-K and VE-gate input probes

    def routed_keys(self, sequence):
        # One sequence's clean keys and values for the own-query response, and for each query its strongest keys (as
        # indices), their probabilities, and the rest of its output, kept for exact renormalization
        d, model = self.capture, self.model
        length, heads = d.length, model.config.n_head
        width = self.width // heads
        cos, sin = model.cos[0, :length, 0].float(), model.sin[0, :length, 0].float()
        pos = torch.arange(length, device=self.device)
        c, s = cos[:, None], sin[:, None]
        inputs, w_q, keys, values, w_proj_t, indices, top_p, rests = [], [], [], [], [], [], [], []
        for i, block in enumerate(model.transformer.h):
            prefix = f'transformer.h.{i}.attn.'
            k = d.outputs[prefix + 'c_k'][sequence].float().view(length, heads, width)
            k1, k2 = k[..., :width // 2], k[..., width // 2:]
            k = norm(torch.cat([k1 * c + k2 * s, k1 * -s + k2 * c], -1))
            v = d.outputs[prefix + 'c_v'][sequence].float().view(length, heads, width)
            if block.attn.ve_gate is not None:
                gate = 2 * torch.sigmoid(d.outputs[prefix + 've_gate'][sequence].float().view(length, heads, 1))
                v = v + gate * d.outputs[f'value_embeds.{i}'][sequence].float().view(length, heads, width)
            x, wq = d.blocks[i][sequence].float(), block.attn.c_q.weight.float()
            kk, vv = k.transpose(0, 1).to(torch.bfloat16).contiguous(), v.transpose(0, 1).to(torch.bfloat16).contiguous()
            r = (norm(x) @ wq.T).view(length, heads, width)
            q = norm(torch.cat([r[..., :width // 2] * c + r[..., width // 2:] * s,
                                r[..., :width // 2] * -s + r[..., width // 2:] * c], -1))
            lag = pos[:, None] - pos[None]
            allowed = (lag >= 0) & (lag <= model.window_sizes[i][0])
            prob = torch.softmax((torch.einsum('thd,hsd->hts', q.to(kk.dtype), kk).float() * width ** -0.5)
                                 .masked_fill(~allowed, float('-inf')), -1)
            p, idx = prob.topk(min(self.s.query_keys, length), dim=-1)
            out = torch.einsum('hts,hsd->htd', prob.to(vv.dtype), vv).float()
            del prob
            head = torch.arange(heads, device=self.device)[:, None, None]
            rest = torch.cat([out[:, part] - torch.einsum('htk,htkd->htd', p[:, part], vv[head, idx[:, part]].float())
                              for part in pos.split(KEY_BUCKET)], 1)
            inputs.append(x)
            w_q.append(wq)
            keys.append(kk)
            values.append(vv)
            w_proj_t.append(block.attn.c_proj.weight.float().T.contiguous())
            indices.append(idx.transpose(0, 1).int().contiguous())
            top_p.append(p.transpose(0, 1).contiguous())
            rests.append(rest.transpose(0, 1).contiguous())
        return pos, inputs, w_q, keys, values, indices, top_p, rests, w_proj_t, (cos, sin)

    def chain_credits(self, e, x0, mids, outs, routes, y):
        # The embedding's and mid streams' credits, measured from the loss along a full orthogonal cycle each, replayed
        # one sequence and one block of tokens at a time. The mixing scalars keep their central mid-stream probes.
        model, s = self.model, self.s
        L = model.config.n_layer
        weight = model.lm_head.weight.float()[:model.config.vocab_size]
        weight_t = weight.T.to(torch.bfloat16).contiguous()
        w_fc = [blk.mlp.c_fc.weight.to(torch.bfloat16) for blk in model.transformer.h]
        w_out_t = [blk.mlp.c_proj.weight.to(torch.bfloat16).T.contiguous() for blk in model.transformer.h]
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        xL = outs[-1]
        base = xL @ weight.T
        target_w = weight[y.clamp(min=0)]
        n, m, h, P = e.shape[0], s.chain_tokens, s.scalar_step, self.width
        probes = []
        for site in range(-1, L):  # the bases are drawn in site order
            H = self.basis()
            probed = e if site < 0 else mids[site]
            steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            probes.append((site, H, steps, torch.empty(n, self.width, device=self.device),
                           torch.zeros(2, device=self.device)))
        length = self.capture.length
        for start in range(0, n, length):
            pos, inputs, w_q, keys, values, indices, top_p, rests, w_proj_t, cos_sin = self.routed_keys(start // length)
            for t0 in range(start, start + length, m):
                t1 = min(t0 + m, start + length)
                sl, local = slice(t0, t1), slice(t0 - start, t1 - start)
                own = (pos[local], [z[local] for z in inputs], w_q, keys, values, [z[local] for z in indices],
                       [z[local] for z in top_p], [z[local] for z in rests], w_proj_t, cos_sin)
                for site, H, steps, credit, lam in probes:
                    move = steps[sl].unsqueeze(-1) * H.unsqueeze(0)
                    scale = steps[sl].expand(-1, P)
                    if site >= 0:  # symmetric probes of both linear mixing scalars
                        prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                        move = torch.cat([move, h * prev, h * x0[sl].unsqueeze(1),
                                          -h * prev, -h * x0[sl].unsqueeze(1)], 1)
                        scale = torch.cat([scale, torch.full_like(scale[:, :2], h),
                                           torch.full_like(scale[:, :2], -h)], 1)
                    derivative = chain_chunk(
                        site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl], [r[0][sl] for r in routes],
                        [r[1][sl] for r in routes], lam_r, lam_0, w_fc, w_out_t, weight_t, base[sl], target_w[sl],
                        y[sl], own)
                    credit[sl] = derivative[:, :P] @ H / self.width
                    if site >= 0:
                        lam += (derivative[:, P:P + 2] + derivative[:, P + 2:]).sum(0) / 2
            del own, inputs, keys, values, indices, top_p, rests
        credits = {}
        for site, _, _, credit, lam in probes:
            credits['emb' if site < 0 else site] = credit
            if site >= 0:
                credits['lam', site] = lam
        return credits

    def key_input_credits(self, targets):
        # Independent central probes at each block input, moving only its K branch, on a private copy of the probe RNG
        # so that they neither shift the Q/hidden probes nor need another RNG stream; with each block's raw-K errors,
        # from the same contexts. Partial Hadamard cycles use randomized rows and column signs.
        d, s = self.capture, self.s
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        credits, key_errors = {}, {}
        for i, target in targets.items():
            indices = torch.randperm(self.width, device=self.device, generator=generator)[:s.key_path_draws]
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            rows = frame[indices] * signs
            estimate = torch.zeros(d.batch, d.length, self.width, device=self.device)
            raw_credit = torch.empty_like(estimate)
            w_key = self.model.transformer.h[i].attn.c_k.weight.float()
            projected_rows = rows @ w_key.T
            window = self.model.window_sizes[i][0]
            for r in range(d.batch):
                context = self.key_context(i, r, target)
                qp, _, raw, cos, sin, A, C = context
                raw_credit[r] = self.key_errors(i, context).transpose(0, 1).reshape(d.length, self.width)
                xin = d.blocks[i][r].float()
                projected_x = xin @ w_key.T
                steps = s.key_path_sigma * xin.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                for t0 in range(0, d.length, s.key_path_tokens):
                    t1 = min(t0 + s.key_path_tokens, d.length)
                    sl = slice(t0, t1)
                    # Queries outside this union of source windows have exactly zero response; the retained range is
                    # rounded to kernel buckets. All directions share a source tile.
                    q1 = min(d.length, t1 + window)
                    qlength = min(d.length, -(-(q1 - t0) // KEY_BUCKET) * KEY_BUCKET)
                    qs = slice(max(0, q1 - qlength), q1)
                    moves = prepared_key_moves(xin[sl], steps[sl], rows, projected_x[sl],
                                               projected_rows, raw[:, sl], cos[sl], sin[sl])
                    scores = key_response(qp[:, qs], moves, A[:, qs, sl], C[:, qs, sl], t0, qs.start, window)
                    estimate[r, sl] = decode_key_pairs(scores, steps[sl], rows)
            credits[i] = estimate.flatten(0, 1)
            key_errors[f'transformer.h.{i}.attn.c_k'] = raw_credit
        return credits, key_errors

    def gate_input_credits(self, targets, credits):
        # independent gate-input experiments from the raw hubs; only linear consumers follow
        d, s = self.capture, self.s
        heads = self.model.config.n_head
        generator = torch.Generator(self.device).set_state(self.generator.get_state())
        frame = hadamard(self.width, self.device, torch.float32)
        gate_credits = {}
        for i in targets:
            # Advance for every layer to retain the tested private pilot bases.
            indices = torch.randperm(self.width, device=self.device, generator=generator)
            signs = torch.randint(0, 2, (self.width,), device=self.device, generator=generator) * 2 - 1
            if not has_ve(i, self.model.config.n_layer):
                continue
            rows = frame[indices] * signs
            value = credits[i].reshape(-1, heads, self.width // heads)
            ve = d.outputs[f'value_embeds.{i}'].float().reshape_as(value)
            kappa = (value * ve).sum(-1)
            xin = d.blocks[i].float().flatten(0, 1)
            weight = self.model.transformer.h[i].attn.ve_gate.weight.float()
            projected_rows = F.linear(rows[:, :weight.shape[1]], weight)
            row_square = rows.square().mean(-1)
            result = torch.empty_like(xin)
            for t0 in range(0, len(xin), s.chain_tokens):
                sl = slice(t0, t0 + s.chain_tokens)
                result[sl] = prepared_gate_input_response(
                    xin[sl], weight, kappa[sl], rows, projected_rows, row_square, s.gate_path_sigma)
            gate_credits[i] = result
        return gate_credits

    def path_credits(self, targets, credits):
        # The input responses of each block: PR #23's value path (the measured value credit through c_v and one
        # RMSNorm step), the input-K probes and the VE-gate input probes, summed; then carried only along the residual
        # identities to the earlier blocks (extra). Also the raw-K errors.
        model, d = self.model, self.capture
        L = model.config.n_layer
        inputs = {i: norm_move(d.blocks[i].float().flatten(0, 1),
                               credits[i].reshape(-1, self.width) @ model.transformer.h[i].attn.c_v.weight.float())
                  for i in range(L)}
        key_credits, key_errors = self.key_input_credits(targets)
        inputs = {i: inputs[i] + key_credits[i] for i in range(L)}
        for i, credit in self.gate_input_credits(targets, credits).items():
            inputs[i] = inputs[i] + credit
        extra = {L - 1: torch.zeros_like(inputs[0])}
        for i in reversed(range(1, L)):
            extra[i - 1] = model.resid_lambdas[i].float() * (inputs[i] + extra[i])
        return inputs, extra, key_errors

    def gradients(self, x, y):
        d, model, s = self.capture, self.model, self.s
        d.capture(x, y)
        L = model.config.n_layer
        b, t = x.shape
        yf = y.flatten()
        gradients = {'lm_head.weight': d.reassemble('lm_head', d.head_error())}
        e = d.outputs['transformer.wte'].float().flatten(0, 1)
        x0 = norm(e)
        mids, outs, routes = [], [], []
        for i in range(L):
            p = f'transformer.h.{i}.'
            mid = d.blocks[i].float().flatten(0, 1) + d.outputs[p + 'attn.c_proj'].float().flatten(0, 1)
            mlp = model.transformer.h[i].mlp
            hpre = F.linear(norm(mid), mlp.c_fc.weight.float())
            r = F.relu(hpre).square()
            out = mid + F.linear(r, mlp.c_proj.weight.float())
            top = (r * mlp.c_proj.weight.float().norm(dim=0)).topk(s.route_k, dim=-1).indices  # the replayed units
            mids.append(mid), outs.append(out), routes.append((top, hpre.gather(1, top)))
        g_top = self.probe(outs[-1], yf)  # the top stream's credit, through the head alone
        chain = self.chain_credits(e, x0, mids, outs, routes, yf)
        lam_r = model.resid_lambdas.float()
        targets = {i: (chain[i] @ block.attn.c_proj.weight.float()).view(b, t, self.width)
                   for i, block in enumerate(model.transformer.h)}
        credits = {i: self.value_credit(i, target) for i, target in targets.items()}  # from the measured targets
        inputs, extra, key_errors = self.path_credits(targets, credits)
        value_targets = {}
        scalars = torch.zeros(2, L, device=self.device)
        for i in range(L):
            p = f'transformer.h.{i}.'
            block = model.transformer.h[i]
            d_out = g_top if i == L - 1 else lam_r[i + 1] * chain[i + 1]
            x_mid, x_out = chain[i] + extra[i], d_out + extra[i]
            gradients[p + 'attn.c_proj.weight'] = x_mid.T @ d.inputs[p + 'attn.c_proj'].flatten(0, 1)
            gradients[p + 'mlp.c_proj.weight'] = x_out.T @ d.inputs[p + 'mlp.c_proj'].flatten(0, 1)
            value_targets[i] = (x_mid @ block.attn.c_proj.weight.float()).view(b, t, self.width)
            u = d.outputs[p + 'mlp.c_fc'].float().flatten(0, 1)
            hidden = self.hidden_credit(u, d_out, block.mlp.c_proj.weight)
            gradients[p + 'mlp.c_fc.weight'] = hidden.T @ d.inputs[p + 'mlp.c_fc'].flatten(0, 1)
            scalars[:, i] = chain['lam', i]
        for i in range(L):  # the mixing scalars' input responses and identity-path credit
            previous = x0 if i == 0 else outs[i - 1]
            credit = inputs[i] + extra[i]
            scalars[:, i] += torch.stack([(credit * previous).sum(), (credit * x0).sum()])
        value_credits = {i: self.value_credit(i, target) for i, target in value_targets.items()}
        self.internals(gradients, targets, key_errors, credits, value_credits)
        gradients['transformer.wte.weight'] = d.reassemble('transformer.wte', chain['emb'].view(b, t, self.width))
        gradients['resid_lambdas'], gradients['x0_lambdas'] = scalars[0], scalars[1]
        # This capture's measured hub targets, never corrected ones; only the source weights' gradients change.
        add_future_gradients(self, gradients, targets)
        return gradients


# -------------------------------------------------------------------------------------------------- future probes


SCORE_COLUMNS = 32  # Smaller tiles avoid shared-memory/occupancy pressure on RTX 5090.


def receiver_changes(x, move, kraw, gate, ve, wkv, wg, cos, sin):  # the finite K and effective-V changes per source
    m, p, width = move.shape
    heads, hd = kraw.shape[-2:]
    dn = norm_delta(x[:, None], move)
    projected = F.linear(dn, wkv).reshape(m, p, 2, heads, hd)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(kraw[:, None], c, s)
    dr = rotate(projected[:, :, 0], c, s)
    dk = norm_delta(r0, dr)
    dv = projected[:, :, 1]
    if gate is not None:
        dg = F.linear(dn[..., :wg.shape[-1]], wg)
        gate_change = 2 * ((gate[:, None] + dg).sigmoid() - gate[:, None].sigmoid())
        dv = dv + gate_change[..., None] * ve[:, None]
    return dk.permute(2, 1, 0, 3).contiguous(), dv.permute(2, 1, 0, 3).contiguous()


@cache
def future_kernels():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def scores(Q, G, DK, DV, A, C, Out,
               T: tl.constexpr, M: tl.constexpr, P: tl.constexpr, D: tl.constexpr,
               QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
               GH: tl.constexpr, GT: tl.constexpr, GD: tl.constexpr,
               AH: tl.constexpr, AT: tl.constexpr, AM: tl.constexpr,
               CH: tl.constexpr, CT: tl.constexpr, CM: tl.constexpr,
               Start, Window: tl.constexpr, BC: tl.constexpr, BQ: tl.constexpr):
        # the exact single-column softmax update of each later query, from each source's K and V changes
        head = tl.program_id(1)
        col = tl.program_id(0) * BC + tl.arange(0, BC)
        source = col % M
        features = tl.arange(0, D)
        offsets = (head * P * M + col[None, :]) * D + features[:, None]
        dk = tl.load(DK + offsets, col[None, :] < P * M, 0)
        dv = tl.load(DV + offsets, col[None, :] < P * M, 0)
        first = (Start + tl.min(source, 0)) // BQ * BQ
        last = tl.minimum(T, Start + tl.max(source, 0) + Window + 1)
        result = tl.full((BC,), 0, tl.float32)
        for start in range(first, last, BQ):
            query = start + tl.arange(0, BQ)
            q = tl.load(Q + head * QH + query[:, None] * QT + features[None, :] * QD,
                        query[:, None] < T, 0)
            g = tl.load(G + head * GH + query[:, None] * GT + features[None, :] * GD,
                        query[:, None] < T, 0)
            ds = tl.dot(q, dk, input_precision='tf32x3') * (D ** -.5)
            vs = tl.dot(g, dv, input_precision='tf32x3')
            valid = (query[:, None] < T) & (col[None, :] < P * M)
            a = tl.load(A + head * AH + query[:, None] * AT + source[None, :] * AM, valid, 0)
            c = tl.load(C + head * CH + query[:, None] * CT + source[None, :] * CM, valid, 0)
            changed = a * libdevice.expm1(ds)
            allowed = (query[:, None] > Start + source[None, :]) & (
                query[:, None] - Start - source[None, :] <= Window)
            response = (changed * c + (a + changed) * vs) / (1 + changed)
            result += tl.sum(tl.where(valid & allowed, response, 0), 0)
        tl.store(Out + head * P * M + col, result, col < P * M)

    @triton.jit
    def query(DQ, K, V, A, Base, Out,
              M: tl.constexpr, P: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              T: tl.constexpr, KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
              VH: tl.constexpr, VT: tl.constexpr, VD: tl.constexpr,
              AH: tl.constexpr, AM: tl.constexpr, AT: tl.constexpr,
              OH: tl.constexpr, OM: tl.constexpr, OD: tl.constexpr,
              BP: tl.constexpr, BK: tl.constexpr):
        # the exact finite softmax response of a moved query, streamed without a token/probe/key tensor
        token, head = tl.program_id(1) // H, tl.program_id(1) % H
        probe = tl.program_id(0) * BP + tl.arange(0, BP)
        feature = tl.arange(0, D)
        delta = tl.load(DQ + ((token * P + probe[:, None]) * H + head) * D + feature[None, :],
                        probe[:, None] < P, 0)
        numerator = tl.full((BP, D), 0, tl.float32)
        partition = tl.full((BP,), 0, tl.float32)
        for start in range(0, T, BK):
            key = start + tl.arange(0, BK)
            a = tl.load(A + head * AH + token * AM + key * AT, key < T, 0)
            # Causal/window-masked tiles are exactly zero, not a top-k truncation.
            if tl.sum(a, 0) > 0:
                k = tl.load(K + head * KH + key[None, :] * KT + feature[:, None] * KD, key[None, :] < T, 0)
                v = tl.load(V + head * VH + key[:, None] * VT + feature[None, :] * VD, key[:, None] < T, 0)
                ds = tl.dot(delta, k, input_precision='tf32x3') * (D ** -.5)
                change = a[None, :] * libdevice.expm1(ds)
                partition += tl.sum(change, 1)
                numerator += tl.dot(change, v, input_precision='tf32x3')
        base = tl.load(Base + head * OH + token * OM + feature * OD)
        response = (numerator - partition[:, None] * base[None, :]) / (1 + partition[:, None])
        tl.store(Out + ((token * P + probe[:, None]) * H + head) * D + feature[None, :],
                 response, probe[:, None] < P)

    return scores, query


@torch.library.custom_op('zotorch_noise::future_scores', mutates_args=())
def future_scores(q: torch.Tensor, target: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor,
                  a: torch.Tensor, contrast: torch.Tensor, source_start: int, window: int) -> torch.Tensor:
    import triton
    h, p, m, d = dk.shape
    out = torch.empty((h, p, m), device=q.device, dtype=torch.float32)
    future_kernels()[0][(triton.cdiv(p * m, SCORE_COLUMNS), h)](
        q, target, dk.contiguous(), dv.contiguous(), a, contrast, out,
        q.shape[1], m, p, d, *q.stride(), *target.stride(),
        *a.stride(), *contrast.stride(), source_start, window, SCORE_COLUMNS, 64,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@future_scores.register_fake
def _(q, target, dk, dv, a, contrast, source_start, window):
    return torch.empty(dk.shape[:-1], device=q.device, dtype=torch.float32)


@torch.library.custom_op('zotorch_noise::future_query_response', mutates_args=())
def query_response(dq: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                   probabilities: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    import triton
    m, p, h, d = dq.shape
    out = torch.empty_like(dq, memory_format=torch.contiguous_format)
    future_kernels()[1][(triton.cdiv(p, 32), m * h)](
        dq.contiguous(), keys, values, probabilities, base, out, m, p, h, d, keys.shape[1],
        *keys.stride(), *values.stride(), *probabilities.stride(), *base.stride(),
        32, 64, num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@query_response.register_fake
def _(dq, keys, values, probabilities, base):
    return torch.empty_like(dq, memory_format=torch.contiguous_format)


def signed_hidden_moves(raw, directions, step, weight):
    # The exact paired squared-ReLU responses for sign directions, in one large GEMM: the even component does not
    # depend on each +/-1 direction. Threshold crossings included; not a derivative approximation.
    base = raw.relu()
    plus, minus = (raw + step).relu(), (raw - step).relu()
    dp, dm = (plus - base) * (plus + base), (minus - base) * (minus + base)
    odd, even = (dp - dm) * .5, (dp + dm) * .5
    moved = F.linear(directions * odd[:, None], weight)
    center = F.linear(even, weight)[:, None]
    return torch.cat((center + moved, center - moved), 1)


def future_embedding_moves(raw, directions, step):
    delta = step[:, None] * directions
    return torch.cat((norm_delta(raw[:, None], delta), norm_delta(raw[:, None], -delta)), 1)


@torch.compile(dynamic=False)
def query_mlp_moves(raw, directions, step, cos, sin, keys, values, probabilities, weight,
                    base, mid, hidden, w_fc, w_out):
    # Each signed Q trial's finite attention output change, carried through its own block's complete MLP, anchored at
    # the captured hidden pre-activation as receiver_changes anchors K/V; ReLU crossings included. The incoming
    # attention and residual move stays part of the block output.
    m, p, width = directions.shape
    heads, _, hd = keys.shape
    delta = torch.cat((step[:, None] * directions, -step[:, None] * directions), 1)
    c, s = cos[:, None, None], sin[:, None, None]
    r0 = rotate(raw.reshape(m, 1, heads, hd), c, s)
    dr = rotate(delta.reshape(m, 2 * p, heads, hd), c, s)
    response = query_response(norm_delta(r0, dr), keys, values, probabilities, base)
    move = F.linear(response.flatten(-2), weight)
    dh = F.linear(norm_delta(mid[:, None], move), w_fc)
    base = hidden[:, None].relu()
    moved = (hidden[:, None] + dh).relu()
    return move + F.linear((moved - base) * (moved + base), w_out)


compiled_hidden = torch.compile(signed_hidden_moves, dynamic=False)
compiled_embedding = torch.compile(future_embedding_moves, dynamic=False)


@torch.compile(dynamic=True)
def receiver_scores(move, coefficient, x, kraw, gate, ve, wkv, wg, cos, sin,
                    q, target, a, contrast, start, window):
    dk, dv = receiver_changes(x, move * coefficient, kraw, gate, ve, wkv, wg, cos, sin)
    return future_scores(q, target, dk, dv, a, contrast, start, window).sum(0).T


class FiniteFuture:
    # One sequence's future probes: central pairs for each source token of the first blocks' hidden and Q outputs and
    # of the embedding, along partial Hadamard cycles, scored at every later block's measured hub target
    def __init__(self, estimator, targets, sequence, seed):
        self.e, self.s, self.seed = estimator, estimator.s, seed
        d, model = estimator.capture, estimator.model
        t, heads = d.length, model.config.n_head
        cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
        self.contexts = []
        for i, block in enumerate(model.transformer.h):
            prefix = f'transformer.h.{i}.attn.'
            qraw, kraw, vraw = [d.outputs[prefix + n][sequence:sequence + 1].float() for n in ('c_q', 'c_k', 'c_v')]
            v = vraw.reshape(1, t, heads, -1)
            gate = ve = wg = None
            if block.attn.ve_gate is not None:
                gate = d.outputs[prefix + 've_gate'][sequence].float()
                ve = d.outputs[f'value_embeds.{i}'][sequence].float().reshape(t, heads, -1)
                wg = block.attn.ve_gate.weight.float()
                v = v + (2 * gate.sigmoid())[None, ..., None] * ve[None]
            values = v.transpose(1, 2)[0]
            q, k, a, out = self.attention(qraw, kraw, values, cos, sin, heads, model.window_sizes[i][0])
            target = targets[i][sequence].reshape(t, heads, -1).transpose(0, 1)
            contrast = (target @ values.transpose(-1, -2)) - (target * out).sum(-1, keepdim=True)
            context = dict(x=d.blocks[i][sequence].float(), q=q, k=k, kraw=kraw[0].reshape(t, heads, -1),
                           values=values, a=a, out=out, contrast=contrast, target=target, gate=gate, ve=ve, wg=wg,
                           wkv=torch.cat((block.attn.c_k.weight, block.attn.c_v.weight)).float(),
                           window=model.window_sizes[i][0])
            if i < self.s.future_layers:
                context['source_mlp'] = (
                    d.blocks[i][sequence].float() + d.outputs[prefix + 'c_proj'][sequence].float(),
                    d.outputs[f'transformer.h.{i}.mlp.c_fc'][sequence].float(),
                    block.mlp.c_fc.weight.float(), block.mlp.c_proj.weight.float())
            self.contexts.append(context)
        self.sequence = sequence
        self.cos, self.sin = cos[0, :, 0], sin[0, :, 0]

    @staticmethod
    def attention(qraw, kraw, values, cos, sin, nhead, window):  # dense clean attention, for independent trials
        def heads(raw):
            shaped = raw.reshape(*raw.shape[:2], nhead, -1)
            return norm(apply_rotary_emb(shaped, cos, sin)).transpose(1, 2)
        q, k = heads(qraw), heads(kraw)
        pos = torch.arange(qraw.shape[1], device=qraw.device)
        lag = pos[:, None] - pos[None, :]
        allowed = (lag >= 0) & (lag <= window)
        a = ((q @ k.transpose(-1, -2)) * q.shape[-1] ** -.5).masked_fill(~allowed, -torch.inf).softmax(-1)
        return q[0], k[0], a[0], (a @ values[None])[0]

    def sources(self):
        for family in ('hidden', 'query'):
            for layer in range(self.s.future_layers):
                yield family, layer, f'transformer.h.{layer}.{"mlp.c_fc" if family == "hidden" else "attn.c_q"}'
        yield 'wte', -1, 'transformer.wte'

    def trial(self, family, layer, raw, sl, receivers, start):
        # the decoded credit of one token chunk's central trials along rows: the source's finite response to each
        # move, scored by every receiver; the clean tensors are bound once per chunk, outside the direction loop
        model = self.e.model
        if family == 'hidden':
            response = partial(compiled_hidden, raw[sl], weight=model.transformer.h[layer].mlp.c_proj.weight.float())
        elif family == 'wte':
            response = partial(compiled_embedding, raw[sl])
        else:
            ctx = self.contexts[layer]
            mid, hidden, w_fc, w_out = ctx['source_mlp']
            response = partial(query_mlp_moves, raw[sl], cos=self.cos[sl], sin=self.sin[sl], keys=ctx['k'],
                               values=ctx['values'], probabilities=ctx['a'][:, sl], base=ctx['out'][:, sl],
                               weight=model.transformer.h[layer].attn.c_proj.weight.float(),
                               mid=mid[sl], hidden=hidden[sl], w_fc=w_fc, w_out=w_out)
        scorers = []
        for j, coefficient in receivers:
            ctx = self.contexts[j]
            scorers.append(partial(receiver_scores, coefficient=coefficient,
                x=ctx['x'][sl], kraw=ctx['kraw'][sl],
                gate=None if ctx['gate'] is None else ctx['gate'][sl],
                ve=None if ctx['ve'] is None else ctx['ve'][sl],
                wkv=ctx['wkv'], wg=ctx['wg'], cos=self.cos[sl], sin=self.sin[sl],
                q=ctx['q'], target=ctx['target'], a=ctx['a'][:, :, sl],
                contrast=ctx['contrast'][:, :, sl], start=start, window=ctx['window']))

        def decoded(rows, steps):
            move = response(rows, steps)
            total = scorers[0](move)
            for receiver in scorers[1:]:
                total = total + receiver(move)
            return decode_local_response(total, steps, rows)
        return decoded

    def credits(self):  # {source site: its tokens' credit}
        e, model, d = self.e, self.e.model, self.e.capture
        draws = self.s.future_draws
        chunk = gcd(self.s.future_chunk, draws)
        lam, lam0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        result = {}
        for family, layer, name in self.sources():
            raw = d.outputs[name][self.sequence].float()
            t, width = raw.shape
            gen = torch.Generator(raw.device).manual_seed(
                self.seed + 7919 * (layer + 2) + 101 * ('hidden query wte'.split().index(family)))
            # Per-token random signs/offsets prevent a shared low-rank projection
            # from persisting in the batch's parameter gradient.
            frame = hadamard(width, raw.device, torch.float32)
            permutation = torch.randperm(width, device=raw.device, generator=gen)
            offsets = torch.randint(width, (t,), device=raw.device, generator=gen)
            signs = 2 * torch.randint(2, (t, width), device=raw.device, generator=gen) - 1
            steps = self.s.future_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            credit = torch.zeros_like(raw)
            receivers = []
            coeff = torch.ones((), device=raw.device)
            if family == 'wte':
                for j in range(len(self.contexts)):
                    coeff = lam[j] * coeff + lam0[j]
                    receivers.append((j, coeff))
            else:
                for j in range(layer + 1, len(self.contexts)):
                    coeff = coeff * lam[j]
                    receivers.append((j, coeff))
            for t0 in range(0, t, self.s.future_tokens):
                sl = slice(t0, min(t0 + self.s.future_tokens, t))
                trial = self.trial(family, layer, raw, sl, receivers, t0)
                accumulator = torch.zeros_like(raw[sl])
                for p0 in range(0, draws, chunk):
                    index = (offsets[sl, None] + permutation[None, p0:p0 + chunk]) % width
                    accumulator += trial(frame[index] * signs[sl, None], steps[sl])
                credit[sl] = accumulator / draws
            result[name] = credit
        return result


def add_future_gradients(estimator, gradients, targets):
    # Adds independent forward responses to the source weights' gradients, one sequence at a time. A fresh frame seed
    # comes from the RNG each update: hidden uses a partial cycle, so a frame fixed across training would bias it
    # toward a persistent subspace.
    s, d = estimator.s, estimator.capture
    seed = int(torch.randint(2 ** 31, (), device=estimator.device, generator=d.generator).item())
    for sequence in range(d.batch):
        for site, credit in FiniteFuture(estimator, targets, sequence, seed + 104729 * sequence).credits().items():
            inputs = d.inputs[site][sequence]
            gradient = gradients[site + '.weight']
            if 'mlp.c_fc' in site:
                raw = d.outputs[site][sequence].float()
                step = s.future_sigma * raw.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                # Coordinates that are inactive in every signed trial have no
                # response; remove their partial-cycle decoding cross-talk.
                credit = credit * (raw + step > 0)
            if site == 'transformer.wte':
                gradient.index_add_(0, inputs.flatten(), credit.flatten(0, -2))
            else:
                gradient.addmm_(credit.T, inputs)
