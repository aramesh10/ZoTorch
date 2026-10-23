import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import zotorch as zt

# The demo's DUST against dust/dust.py with shared noise: each ZoTorch estimator runs with its seeds recorded, then
# dust.py's matching function runs with DUST.noise returning that same noise, in the order and layout dust.py asks
# for it. With the same noise, the two must give the same weight gradients. On the CPU, in float32.

ROOT = Path(__file__).resolve().parents[1]
DUST = ROOT.parent / 'dust'
if not (DUST / 'dust.py').exists():
    pytest.skip('needs dust/dust.py beside this repository', allow_module_level=True)
sys.path.insert(0, str(ROOT / 'demo'))
from model import gpt  # noqa: E402
from optimizers.dust import DustDirect, DustHead, DustLocal  # noqa: E402
from optimizers.mezo import MeZO  # noqa: E402


def reference(name):  # dust/<name>.py, whose `from model import` must find dust/model.py rather than demo/model
    spec = importlib.util.spec_from_file_location(f'dust_{name}', DUST / f'{name}.py')
    module, ours = importlib.util.module_from_spec(spec), sys.modules['model']
    sys.modules['model'] = sys.modules.get('dust_model', ours)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules['model'] = ours
    sys.modules[spec.name] = module
    return module


reference_model, reference_dust = reference('model'), reference('dust')
B, T, NREP = 2, 64, 2
KINDS = {'q': 'c_q', 'v': 'c_v', 'gate': 've_gate', 'k': 'c_k', 've': 've'}  # dust.py's kinds, by our names


class Recorder:  # an optimizer that keeps the estimated gradients
    def __init__(self):
        self.gradients = {}

    def accumulate(self, param, grad):
        self.gradients[param] = self.gradients.get(param, 0) + grad.numpy()


def z(seed, *shape):  # zt.noise(seed) on the CPU
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def normal_(weight, seed, std):
    weight.normal_(0.0, std, generator=torch.Generator().manual_seed(seed))


@pytest.fixture
def setup(monkeypatch):
    config = gpt.GPTConfig(sequence_len=T, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
    ours = gpt.make_model(config, 'cpu', seed=0)
    normal_(ours.lm_head.weight, 1000, std=0.5)  # large enough that the loss depends on every layer
    normal_(ours.value_embeds['1'].weight, 1001, std=0.5)
    normal_(ours.transformer.h[1].attn.ve_gate.weight, 1002, std=0.5)

    theirs = reference_model.GPT(reference_model.GPTConfig(**vars(config)))
    theirs.init_weights()  # for its rotary tables
    theirs.load_state_dict({name: p.clone() for name, p in ours.named_parameters()})
    theirs.requires_grad_(False)

    tokens = torch.randint(0, config.vocab_size, (B, T + 1), generator=torch.Generator().manual_seed(0))
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    recipe = reference_dust.Recipe(writers=[2, 2], hidden=[2, 2], hubs=[2, 2], embedding=2, head=8,
                                   local={kind: 2 for kind in KINDS}, nrep=NREP, lr=1.0, momentum=0.0)
    est = reference_dust.DUST(theirs, recipe)
    with torch.no_grad():
        est.capture(x, y)

    seeds = []
    draw = zt.seed
    monkeypatch.setattr(zt, 'seed', lambda: seeds.append(draw()) or seeds[-1])

    queue = []
    def noise(self, *shape, dtype=None):  # dust.py's noise, taken from the queue
        assert tuple(queue[0].shape) == shape, (tuple(queue[0].shape), shape)
        return queue.pop(0)
    monkeypatch.setattr(reference_dust.DUST, 'noise', noise)

    yield ours, est, (x, y), seeds, queue
    assert not queue  # dust.py used all of the shared noise
    est.close()


def ours(estimator, model, batch):
    opt = Recorder()
    estimator(model, batch, opt)
    return opt.gradients


def same(ours, theirs):  # dust.py sums the loss over tokens and divides by their count at the end
    theirs = (theirs / (B * T)).numpy()
    torch.testing.assert_close(torch.from_numpy(ours), torch.from_numpy(theirs), rtol=1e-3,
                               atol=1e-4 * abs(theirs).max())


@pytest.mark.parametrize('names', [
    ['transformer.h.0.attn.c_proj', 'transformer.h.0.mlp.c_proj'],  # the writers, jittered together in each draw
    ['transformer.h.1.mlp.c_fc'],
    ['transformer.wte'],
])
def test_direct(setup, names):
    model, est, batch, seeds, queue = setup
    modules, draws, k = dict(model.named_modules()), 4, len(names)
    gradients = ours(DustDirect(model, [f'{name}.weight' for name in names], draws=draws), model, batch)

    # Ours: each draw's seeds, one per site. dust.py: each chunk's noise, site by site, its copies stacked.
    for chunk in range(draws // NREP):
        for i, name in enumerate(names):
            width = est.outputs[name].shape[-1]
            queue.append(torch.cat([z(seeds[(chunk * NREP + j) * k + i], B, T, width) for j in range(NREP)]))
    with torch.no_grad():
        errors = est.direct_errors(names, draws, 0.2)
    for name in names:
        same(gradients[modules[name].weight], est.reassemble(name, errors[name]))


def test_head(setup):
    model, est, batch, seeds, queue = setup
    draws = 8
    gradients = ours(DustHead(model, 'lm_head.weight', softcap=15, draws=draws), model, batch)

    for chunk in range(draws // NREP):
        queue.append(torch.cat([z(seeds[chunk * NREP + j], B, T, 256) for j in range(NREP)]))
    with torch.no_grad():
        error = est.head_error(draws)
    same(gradients[model.lm_head.weight], est.reassemble('lm_head', error))


def test_scalars(setup):
    model, est, batch, seeds, queue = setup
    draws, layers = 8, len(model.transformer.h)
    gradients = ours(MeZO(model, ['resid_lambdas', 'x0_lambdas'], eps=0.03, n=draws), model, batch)

    # Both scalars move together in each pair, each with its own seed.
    for resid, x0 in zip(seeds[0::2], seeds[1::2]):
        queue.append(torch.stack([z(resid, layers), z(x0, layers)]))
    with torch.no_grad():
        estimate = est.scalar_gradients(draws)
    for name in ['resid_lambdas', 'x0_lambdas']:
        same(gradients[getattr(model, name)], estimate[name])


def test_local(setup):
    model, est, batch, seeds, queue = setup
    blocks, draws, cfg = len(model.transformer.h), 2, model.config

    # Ours: a DustLocal per block, with each block's seeds: the targets' first, then each kind's in turn.
    gradients, kinds_seeds, target_seeds = {}, {}, {}
    for i in range(blocks):
        attn = f'transformer.h.{i}.attn'
        layers = {'c_q': f'{attn}.c_q.weight', 'c_k': f'{attn}.c_k.weight', 'c_v': f'{attn}.c_v.weight'}
        arguments = {}
        if str(i) in model.value_embeds:  # the value embedding enters the attention as its argument 1
            layers['ve_gate'] = f'{attn}.ve_gate.weight'
            arguments['ve'] = (1, f'value_embeds.{i}.weight')
        estimator = DustLocal(model, attn, f'{attn}.c_proj.weight', layers, arguments,
                              draws={kind: draws for kind in KINDS.values()}, heads=cfg.n_head, target_draws=2,
                              target_sigma=0.2 if i < blocks // 2 else 0.4, window=model.window_sizes[i][0],
                              uncredited=['c_q'], two_sided=['ve_gate'])
        start = len(seeds)
        gradients.update(ours(estimator, model, batch))
        block_seeds = seeds[start:]
        target_seeds[i], block_seeds = block_seeds[:2], block_seeds[2:]
        for kind in estimator.targets[0].weights:
            kinds_seeds[kind, i], block_seeds = block_seeds[:draws], block_seeds[draws:]

    # dust.py: the targets block by block, then each kind across the blocks, in its own order.
    for i in range(blocks):
        queue.append(torch.cat([z(s, B, T, cfg.n_embd) for s in target_seeds[i]]))
    for kind, name in KINDS.items():
        width = cfg.n_head if kind == 'gate' else cfg.n_embd
        for i in range(blocks):
            if (name, i) in kinds_seeds:
                queue.append(torch.stack([z(s, B, T, width) for s in kinds_seeds[name, i]]))

    with torch.no_grad():
        targets = {i: est.direct_errors([f'o@{i}'], 2, 0.2 if i < blocks // 2 else 0.4)[f'o@{i}']
                   for i in range(blocks)}
        for kind in KINDS:
            for site, error in est.local_errors(kind, draws, targets).items():
                same(gradients[model.get_submodule(site).weight], est.reassemble(site, error))
