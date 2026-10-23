import argparse
from contextlib import nullcontext
import time
from pathlib import Path

import torch

import zotorch as zt
from model.gpt import GPTConfig, make_model
from optimizers import dust, dust_optimized
from optimizers.mezo import MeZO

# Training with ZoTorch: a model, a list of estimators and an optimizer. The estimators are the only thing that
# changes between methods.
#
#     python demo/train.py --estimator mezo             SPSA on every weight
#     python demo/train.py --estimator dust             DUST, an estimator per kind of layer
#     python demo/train.py --estimator dust-optimized   DUST with one draw per pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator', choices=['dust', 'dust-optimized', 'mezo'], default='dust')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--data', type=Path, default=Path('/data'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Dataset: rows of tokens; each row is an input and, shifted by one, its targets
    data = {}
    for name, count in [('train_1M', args.steps * args.batch), ('val', 32)]:
        rows = torch.load(args.data / f'{name}.pt', map_location='cpu', weights_only=True)
        if isinstance(rows, dict):  # dust/prepare_data.py's chunked format
            rows = torch.cat([
                chunk.view(rows['batch_size'], rows['sequence_size'])[:valid]
                for chunk, valid in zip(rows['chunks'], rows['valid_counts'])
            ])
        data[name] = rows[:count].long().to(args.device)

    # Model; on the GPU its embeddings are bfloat16, so its passes run under autocast, as in dust.py
    model = make_model(GPTConfig(), args.device, seed=42)
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if args.device.startswith('cuda') else nullcontext()
    for module in [*model.transformer.h, model.lm_head]:  # clean passes rerun only the blocks they capture
        zt.CacheHook(module).register()

    # Estimators and optimizer
    if args.estimator == 'mezo':
        estimators = [MeZO(model, [name for name, _ in model.named_parameters()], eps=1e-3, n=8)]
        opt = zt.optim.SGD({'all': model.parameters()}, lr=1e-3, momentum=0.9)
    else:
        method = dust_optimized if args.estimator == 'dust-optimized' else dust
        # dust.py's recipe for a population of 256, as draws per block
        population = 256
        writers = [38, 32, 16, 10, 6, 4, 2, 2]  # the attention and MLP output projections (c_proj), together
        hidden = [38, 32, 16, 10, 6, 4, 2, 2]   # the MLP's first layer (mlp.c_fc)
        hubs = [6, 6, 4, 2, 2, 2, 2, 2]         # the attention output, before attn.c_proj
        local = {'c_q': 2, 'c_v': 20, 've_gate': 6, 'c_k': 12, 've': 8}  # each attention internal

        estimators = [
            method.DustDirect(model, ['transformer.wte.weight'], draws=10),
            method.DustHead(model, 'lm_head.weight', softcap=15, draws=1024),
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
                method.DustDirect(model, [f'{block}.attn.c_proj.weight', f'{block}.mlp.c_proj.weight'],
                                  draws=writers[i]),
                method.DustDirect(model, [f'{block}.mlp.c_fc.weight'], draws=hidden[i]),
                method.DustLocal(model, attn, f'{attn}.c_proj.weight', layers, arguments, draws=local,
                                 heads=model.config.n_head, target_draws=hubs[i],
                                 target_sigma=0.2 if i < blocks // 2 else 0.4, window=model.window_sizes[i][0],
                                 uncredited=['c_q'], two_sided=['ve_gate']),
            ]

        # dust.py's learning rates, by group of weights
        groups = {
            'wte': [model.transformer.wte.weight],
            'value_embeds': [embedding.weight for embedding in model.value_embeds.values()],
            'lambdas': [model.resid_lambdas, model.x0_lambdas],
        }
        grouped = {id(param) for params in groups.values() for param in params}
        groups['rest'] = [param for param in model.parameters() if id(param) not in grouped]
        lr = {'wte': 1000.0, 'value_embeds': 0.3, 'lambdas': 0.03, 'rest': 0.25}
        opt = zt.optim.SGD(groups, lr=lr, momentum=0.95)

    # Training
    for step in range(args.steps):
        rows = data['train_1M'][step * args.batch:(step + 1) * args.batch]
        x, y = rows[:, :-1], rows[:, 1:]

        start = time.perf_counter()
        with autocast, torch.no_grad():
            loss = model(x, y).mean().item()
            for estimator in estimators:
                estimator(model, (x, y), opt)
        opt.apply()

        print(f'step {step + 1}: loss {loss:.4f} in {time.perf_counter() - start:.1f}s', flush=True)

    # Validation
    rows = data['val']
    with autocast, torch.no_grad():
        loss = model(rows[:, :-1], rows[:, 1:]).mean().item()
    print(f'validation loss {loss:.4f}')
