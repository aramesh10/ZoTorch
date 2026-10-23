import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'demo'))
from model import gpt  # noqa: E402
from optimizers.dust import DustDirect, DustHead, DustLocal  # noqa: E402
from optimizers.mezo import MeZO  # noqa: E402

REFERENCE = ROOT.parent / 'dust' / 'model.py'


class Recorder:  # an optimizer that keeps the estimated gradients
    def __init__(self):
        self.gradients = {}

    def accumulate(self, param, grad):
        self.gradients[param] = self.gradients.get(param, 0) + grad


def batch(config, device):
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, config.vocab_size, (2, config.sequence_len + 1), generator=generator)
    return tokens[:, :-1].contiguous().to(device), tokens[:, 1:].contiguous().to(device)


@pytest.mark.skipif(not REFERENCE.exists(), reason='needs dust/model.py beside this repository')
def test_per_token_losses_match_dust():
    spec = importlib.util.spec_from_file_location('dust_model', REFERENCE)  # not demo/model
    reference_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_model)
    config = gpt.GPTConfig(sequence_len=128, vocab_size=500, n_layer=4, n_head=2, n_kv_head=2, n_embd=128)
    model = gpt.make_model(config, 'cpu', seed=0)
    reference = reference_model.GPT(reference_model.GPTConfig(**vars(config)))
    reference.init_weights()  # for its rotary tables
    reference.load_state_dict({name: p.clone() for name, p in model.named_parameters()})
    x, y = batch(config, 'cpu')
    with torch.no_grad():
        expected = reference(x, y, loss_reduction='none').view(2, -1)
        torch.testing.assert_close(model(x, y), expected)


def test_dust_estimates_every_weight(device):  # every kind of weight in the GPT, with few draws
    config = gpt.GPTConfig(sequence_len=64, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)
    model = gpt.make_model(config, device, seed=0)
    # train.py's DUST estimators, with two draws each
    local = {'c_q': 2, 'c_v': 2, 've_gate': 2, 'c_k': 2, 've': 2}
    estimators = [
        DustDirect(model, ['transformer.wte.weight'], draws=2),
        DustHead(model, 'lm_head.weight', softcap=15, draws=4),
        MeZO(model, ['resid_lambdas', 'x0_lambdas'], eps=0.03, n=8),
    ]
    blocks = len(model.transformer.h)
    for i in range(blocks):
        block = f'transformer.h.{i}'
        attn = f'{block}.attn'
        layers = {'c_q': f'{attn}.c_q.weight', 'c_k': f'{attn}.c_k.weight', 'c_v': f'{attn}.c_v.weight'}
        arguments = {}
        if str(i) in model.value_embeds:  # the value embedding enters the attention as its argument 1
            layers['ve_gate'] = f'{attn}.ve_gate.weight'
            arguments['ve'] = (1, f'value_embeds.{i}.weight')
        estimators += [
            DustDirect(model, [f'{block}.attn.c_proj.weight', f'{block}.mlp.c_proj.weight'], draws=2),
            DustDirect(model, [f'{block}.mlp.c_fc.weight'], draws=2),
            DustLocal(model, attn, f'{attn}.c_proj.weight', layers, arguments, draws=local,
                      heads=model.config.n_head, target_draws=2, target_sigma=0.2 if i < blocks // 2 else 0.4,
                      window=model.window_sizes[i][0], uncredited=['c_q'], two_sided=['ve_gate']),
        ]
    recorder = Recorder()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        for estimator in estimators:
            estimator(model, batch(config, device), recorder)
    for name, param in model.named_parameters():
        grad = recorder.gradients.get(param)
        assert grad is not None and torch.isfinite(grad).all() and grad.any(), name
