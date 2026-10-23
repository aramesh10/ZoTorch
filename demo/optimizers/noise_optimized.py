from dataclasses import dataclass

from optimizers import noise

# dust/dust_noise_optimized.py: noise.py, which keeps its memory savings that leave the estimates unchanged, with the
# two that are settings. The replay runs 256 tokens at a time instead of 512, so its moved streams and its readout
# logits are half the size; this changes the shapes of its bfloat16 matrix products, so its estimates differ from
# noise.py's in rounding (about 1e-5 relative). Boost moves the embedding 4 directions at a time instead of 16, so the
# first attention stage's readout never holds the full float32 logits of 16 moved copies of the batch.


@dataclass
class Settings(noise.Settings):
    chain_tokens: int = 256
    emb_chunk: int = 4


Boost, Cut = noise.Boost, noise.Cut
