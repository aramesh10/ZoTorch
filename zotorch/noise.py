import torch

class noise:
    # scale * coeff * z(seed), standing in for a tensor in a PyTorch operation: z is drawn there, shaped like the
    # operation's tensor, so it is never stored, and the same seed draws the same z for the same number of elements.
    # A tensor coeff is repeated over the dimensions after its own, so per-token rewards scale noise over channels.
    def __init__(self, seed, scale=1.0, coeff=None):
        self.seed, self.scale, self.coeff = seed, scale, coeff

    def __mul__(self, other):
        if isinstance(other, torch.Tensor):
            return noise(self.seed, self.scale, other if self.coeff is None else self.coeff * other)
        return noise(self.seed, self.scale * other, self.coeff)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (1 / other)

    def __neg__(self):
        return self * -1.0

    def like(self, x):  # the noise with x's shape and device, drawn in float32 so every dtype sees the same z
        generator = torch.Generator(x.device).manual_seed(self.seed)
        z = torch.randn(x.shape, generator=generator, device=x.device).mul_(self.scale)
        if self.coeff is not None:
            z.mul_(self.coeff.reshape(*self.coeff.shape, *[1] * (x.ndim - self.coeff.ndim)))
        return z.to(x.dtype)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        x = next(arg for arg in args if isinstance(arg, torch.Tensor))
        return func(*[arg.like(x) if isinstance(arg, noise) else arg for arg in args], **(kwargs or {}))
