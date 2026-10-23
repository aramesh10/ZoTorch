import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import zotorch as zt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demo'))
from optimizers.dust import DustDirect  # noqa: E402
from optimizers.mezo import MeZO  # noqa: E402


class Regression(nn.Module):  # per-token squared error of a linear map, whose gradient is known exactly
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 3, bias=False)

    def forward(self, x, y):
        return (self.layer(x) - y).square().sum(-1)


class Recorder:  # an optimizer that keeps the estimated gradients
    def __init__(self):
        self.gradients = {}

    def accumulate(self, param, grad):
        self.gradients[param] = self.gradients.get(param, 0) + grad


@pytest.fixture
def problem(device):
    generator = torch.Generator().manual_seed(0)
    x, y = torch.randn(4, 8, 4, generator=generator), torch.randn(4, 8, 3, generator=generator)
    model = Regression().to(device).requires_grad_(False)
    model.layer.weight.normal_(generator=torch.Generator(device).manual_seed(1))
    w = model.layer.weight.cpu()
    gradient = 2 * torch.einsum('bto,bti->oi', x @ w.T - y, x) / (4 * 8)  # of the mean per-token loss
    return model, (x.to(device), y.to(device)), gradient


def cosine(a, b):
    return (a * b).sum() / a.norm() / b.norm()


@pytest.mark.parametrize('make', [
    lambda model: MeZO(model, ['layer.weight'], eps=1e-2, n=1000),
    # rewarded against the clean loss: centering within chunks of nrep scales the estimate by 1 - 1/nrep
    lambda model: DustDirect(model, ['layer.weight'], sigma=0.1, draws=1000, nrep=1),
])
def test_gradient(problem, make):
    model, batch, expected = problem
    opt = Recorder()
    make(model)(model, batch, opt)
    estimate = opt.gradients[model.layer.weight].cpu()
    assert cosine(estimate, expected) > 0.95 and 0.8 < estimate.norm() / expected.norm() < 1.2


@pytest.mark.parametrize('make', [
    lambda model: MeZO(model, ['layer.weight'], eps=1e-2, n=16),
    lambda model: DustDirect(model, ['layer.weight'], sigma=0.1, draws=32),
])
def test_training(problem, make):
    model, batch, _ = problem
    estimator = make(model)
    weight, first = model.layer.weight.clone(), model(*batch).mean()
    opt = zt.optim.SGD({'all': model.parameters()}, lr=0.05, momentum=0.5)
    for _ in range(40):
        estimator(model, batch, opt)
        opt.apply()
    assert model(*batch).mean() < 0.5 * first and not torch.equal(model.layer.weight, weight)


def test_noise_is_drawn_again_from_its_seed(device):
    x = torch.zeros(2, 3, 4, device=device)
    pointer = x.data_ptr()
    x += zt.noise(5) * torch.ones(2, device=device) * 3.0
    assert x.data_ptr() == pointer  # in place
    x -= zt.noise(5) * 3.0
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)
    flat = torch.zeros(24, device=device) + zt.noise(9)
    assert torch.equal(flat.view(2, 3, 4), torch.zeros(2, 3, 4, device=device) + zt.noise(9))  # the same for any shape


def test_writer_changes_the_model_only_inside_perturb(problem):
    model, batch, _ = problem
    clean = model(*batch)
    hook = zt.WriterHook(zt.Output(model.layer))
    hook.register()
    assert torch.equal(model(*batch), clean)
    with hook.perturb(lambda value: value + zt.noise(7) * 0.5):
        assert not torch.allclose(model(*batch), clean)
    assert torch.equal(model(*batch), clean)
    hook.remove()


def test_weight_writer_changes_weights_only_during_the_pass(problem):
    model, batch, _ = problem
    weight, clean = model.layer.weight.clone(), model(*batch)
    hook = zt.WriterHook(zt.Weight(model, [model.layer.weight]))
    hook.register()

    def change(params):
        params[0] += zt.noise(7) * 0.5
        return lambda: params[0].sub_(zt.noise(7).like(params[0]) * 0.5)
    with hook.perturb(change):
        assert not torch.allclose(model(*batch), clean)
    torch.testing.assert_close(model.layer.weight, weight)
    hook.remove()


def test_reader_sees_inputs_and_outputs_inside_a_capture(problem):
    model, batch, _ = problem
    inputs, outputs = zt.ReaderHook(zt.Input(model.layer, 0)), zt.ReaderHook(zt.Output(model.layer))
    inputs.register()
    outputs.register()
    model(*batch)
    assert inputs.value is None and outputs.value is None
    with zt.capture():
        model(*batch)
    assert torch.equal(inputs.value, batch[0])
    torch.testing.assert_close(outputs.value, model.layer(batch[0]))
    inputs.remove()
    outputs.remove()


def test_estimator_removes_its_hooks(problem):
    model, batch, _ = problem
    DustDirect(model, ['layer.weight'], draws=4)(model, batch, Recorder())
    assert not model.layer._forward_pre_hooks and not model.layer._forward_hooks
