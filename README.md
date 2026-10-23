# ZoTorch2

Zeroth-order training for any PyTorch `nn.Module`: gradients estimated from forward passes alone, by changing values
inside the model and watching the loss.

```python
import zotorch as zt

estimators = [MeZO(model, [name for name, _ in model.named_parameters()], eps=1e-3, n=8)]
opt = zt.optim.SGD({'all': model.parameters()}, lr=1e-3, momentum=0.9)

for batch in data:
    for estimator in estimators:
        estimator(model, batch, opt)
    opt.apply()
```

The model's one requirement: `model(*batch)` returns per-token losses, `[batch, tokens]`.

## Demos: `demo/`

| | |
|---|---|
| `optimizers/mezo.py` | [MeZO](https://arxiv.org/abs/2305.17333) |
| `optimizers/dust.py` | Dust |
| `optimizers/dust_optimized.py` | Dust with optimizations |
| `optimizers/noise.py` | Noise record |
| `optimizers/noise_optimized.py` | Noise record with optimizations |
| `model/gpt.py` | GPT model |
| `train.py` | training with `--estimator mezo`, `dust` or `dust-optimized` |
| `train_noise.py` | training with `--estimator noise` or `noise-optimized`, with the record's optimizer, schedule and end merge |

## Training

The data is the dust repository's: `train_1M.pt`, `val.pt` and `test.pt` from its `prepare_data.py`.

In Python, from the repository root:

```bash
export PYTHONPATH=.
python demo/train.py --estimator mezo --data <dir>                                     # MeZO
python demo/train.py --estimator dust --data <dir>                                     # Dust
python demo/train.py --estimator dust-optimized --data <dir>                           # Dust with optimizations
python demo/train_noise.py --data <dir> --output runs/noise                            # Noise record
python demo/train_noise.py --estimator noise-optimized --data <dir> --output runs/noise-opt   # with optimizations
```

On Modal, one H100, with the dust repository's `dust-data` volume (runs under `/runs` are kept in `zotorch-runs`):

```bash
modal run modal_app.py --args "--estimator mezo"                                       # MeZO
modal run modal_app.py --args "--estimator dust"                                       # Dust
modal run modal_app.py --args "--estimator dust-optimized"                             # Dust with optimizations
modal run modal_app.py --script train_noise.py --args "--output /runs/noise"           # Noise record
modal run modal_app.py --script train_noise.py --args "--estimator noise-optimized --output /runs/noise-opt"
```

`train.py` takes `--steps` and `--batch`; 
`train_noise.py` takes `--tokens 1M|10M`, `--steps` and `--eval-every`

## Library: `zotorch/`

| | |
|---|---|
| `zt.Output(module)`, `zt.Input(module, i)`, `zt.Weight(module, params)` | sites: where a hook reads or writes |
| `zt.ReaderHook(site)` | records the site's value on passes inside `with zt.capture():` |
| `zt.WriterHook(site)` | changes the site's value on passes inside `with writer.perturb(change):` |
| `zt.perturb({writer: change})` | perturbs several writers at once |
| `zt.CacheHook(module)` | reuses the module's last clean output when nothing about it has changed |
| `zt.ZoWeight(model, name)` | a weight, with readers and writers on its module |
| `zt.ZoModule(model, name, read, layers, arguments)` | a module to rerun alone: `module.replay()` |
| `zt.Estimator(targets)` | runs `estimate(model, batch, opt, target)` per target, or once for all with `__call__ = zt.Estimator.joint` |
| `zt.noise(seed)` | noise never stored: the same seed draws it again |
| `zt.weight_gradient(layer, input, error)` | a `Linear`'s or `Embedding`'s weight gradient from its output error |
| `zt.optim.SGD` | `accumulate(param, grad)`, then `apply()` once per step |

## Tests

`python -m pytest tests`. The `*_exact` tests check the demos against the `dust` repository beside this one; the noise
ones need a CUDA GPU.
