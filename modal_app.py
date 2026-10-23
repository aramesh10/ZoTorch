import os
import subprocess

import modal

# Training on one Modal H100, on the dust repository's data volume (dust: modal run modal_app.py::prepare). Runs written
# under /runs persist in the zotorch-runs volume.
#
#     modal run modal_app.py --script train.py --args "--estimator dust"
#     modal run modal_app.py --script train_noise.py --args "--tokens 1M --output /runs/noise-1m"

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu128')
    .add_local_dir('.', remote_path='/root/ZoTorch2', ignore=['.git', '**/__pycache__', '.pytest_cache', 'runs'])
)
data = modal.Volume.from_name('dust-data')
runs = modal.Volume.from_name('zotorch-runs', create_if_missing=True)
app = modal.App('zotorch', image=image)


@app.function(volumes={'/data': data, '/runs': runs}, gpu='H100', timeout=24 * 60 * 60)
def train(script: str, args: list[str]):
    try:
        subprocess.run(['python', f'demo/{script}', '--data', '/data', *args], cwd='/root/ZoTorch2', check=True,
                       env={**os.environ, 'PYTHONPATH': '/root/ZoTorch2'})
    finally:
        runs.commit()


@app.local_entrypoint()
def main(script: str = 'train.py', args: str = ''):
    train.remote(script, args.split())
