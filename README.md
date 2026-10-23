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
| `optimizers/mezo.py` | MeZO |
| `optimizers/dust.py` | [DUST](https://qlabs.sh/research/dust), as `dust/dust.py` |
| `optimizers/dust_optimized.py` | DUST with one draw per pass, as `dust/dust_optimized.py` |
| `optimizers/noise.py` | the qlabs-eng/noise record, as `dust/dust_noise.py` |
| `optimizers/noise_optimized.py` | the noise record's lower-memory settings, as `dust/dust_noise_optimized.py` |
| `model/gpt.py` | the GPT from `dust/model.py` |
| `train.py` | training with `--estimator mezo`, `dust` or `dust-optimized` |
| `train_noise.py` | training with `--estimator noise` or `noise-optimized`, with the record's optimizer, schedule and end merge |

Run from the repository root with `PYTHONPATH=.`, e.g. `python demo/train.py --estimator dust --data <dir>`.

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
