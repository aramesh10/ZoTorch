import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# demo/optimizers/noise.py against dust/dust_noise.py, and noise_optimized.py against dust/dust_noise_optimized.py: the
# same model, batch and seed must give the same weight gradients from each phase's estimator. On a CUDA GPU, where both
# take the record's compiled paths. Both sides switch their model's attention to compiled flex attention when imported,
# so they are imported only when these tests run.

ROOT = Path(__file__).resolve().parents[1]
DUST = ROOT.parent / 'dust'
if not (DUST / 'dust_noise_optimized.py').exists() or not torch.cuda.is_available():
    pytest.skip('needs dust/dust_noise_optimized.py beside this repository and a CUDA GPU', allow_module_level=True)
sys.path.insert(0, str(ROOT / 'demo'))
PAIRS = {'noise': 'dust_noise', 'noise_optimized': 'dust_noise_optimized'}


def load(name, path):  # registered under name, where torch.compile looks modules up
    spec = importlib.util.spec_from_file_location(name, path)
    module = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def references():
    # dust/dust_noise.py and dust_noise_optimized.py, whose imports of model, dust, dust_noise and stages must find the
    # dust repository's
    ours = sys.modules.get('model')
    sys.modules['model'] = load('dust_model', DUST / 'model.py')
    sys.modules['dust'] = load('dust_dust', DUST / 'dust.py')
    sys.path.append(str(DUST))
    try:
        load('dust_noise', DUST / 'dust_noise.py')
        return {name: importlib.import_module(name) for name in PAIRS.values()}
    finally:
        sys.path.remove(str(DUST))
        sys.modules['model'] = ours


@pytest.fixture(scope='module')
def modules():
    import train_noise
    ours = {name: importlib.import_module(f'optimizers.{name}') for name in PAIRS}
    return ours, references(), train_noise


class Recorder:  # an optimizer that keeps the estimated gradients
    def __init__(self):
        self.gradients = {}

    def accumulate(self, param, grad):
        self.gradients[param] = self.gradients.get(param, 0) + grad


@pytest.mark.parametrize('name', PAIRS)
@pytest.mark.parametrize('phase', ['Boost', 'Cut'])
def test_phase(modules, name, phase):
    ours, theirs, train_noise = modules
    ours, theirs, make_model = ours[name], theirs[PAIRS[name]], theirs['dust_noise'].make_model
    torch.set_float32_matmul_precision('high')
    settings = ours.Settings()
    model = train_noise.make_model(torch.device('cuda'), 0, settings)
    reference_model = make_model(torch.device('cuda'), 0, settings.model, settings.projection_std, settings.head_std)
    reference_model.load_state_dict(model.state_dict())
    tokens = torch.randint(0, settings.model.vocab_size, (settings.cut_batch, settings.model.sequence_len + 1),
                           generator=torch.Generator().manual_seed(0)).cuda()
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    opt = Recorder()
    getattr(ours, phase)(model, settings, 0)(model, (x, y), opt)
    with torch.no_grad():
        expected = getattr(theirs, phase)(reference_model, theirs.Settings(), 0, 0, 1).estimate(x, y)

    valid = (y != -1).sum()
    for parameter, param in model.named_parameters():
        want = expected[parameter] / valid
        torch.testing.assert_close(opt.gradients[param], want, rtol=1e-3, atol=1e-3 * want.abs().max().item(),
                                   msg=lambda m: f'{parameter}: {m}')
