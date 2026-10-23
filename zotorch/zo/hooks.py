from contextlib import contextmanager

import torch

capturing = False
readers = set()     # registered ReaderHooks
perturbing = set()  # WriterHooks inside a perturb
changed = set()     # WriterHooks that have changed a value inside their perturb

@contextmanager
def capture():  # readers record the passes run inside, and keep those values until the next capture
    global capturing
    capturing = True
    yield
    capturing = False

@contextmanager
def perturb(changes):  # {writer: change}: the passes run inside see every writer's change
    for writer, change in changes.items():
        writer.change = change
        perturbing.add(writer)
    yield
    for writer in changes:
        perturbing.discard(writer)
        changed.discard(writer)
        writer.change = None

class Hook:
    def __init__(self, site):
        self.site = site

    def remove(self):
        for handle in self.handles:
            handle.remove()

class WriterHook(Hook):  # while registered, the model sees change(value) at the site; a change of None leaves it
    change = None

    def register(self):
        self.handles = self.site.write(self.write)

    def write(self, value):
        if self.change is not None:
            changed.add(self)
            return self.change(value)

    def perturb(self, change):  # the passes run inside see change
        return perturb({self: change})

    def remove(self):
        super().remove()
        self.change = None

class ReaderHook(Hook):  # while registered, value is the value at the site on the last pass run inside a capture
    value = None

    def register(self):
        self.handles = self.site.read(self.see)
        readers.add(self)

    def see(self, get):
        if capturing:
            self.value = get()

    def remove(self):
        super().remove()
        readers.discard(self)
        self.value = None

class CacheHook:
    # While registered, the module returns its output from the last clean pass (one with no writer perturbing) when its
    # arguments and weights are the same, and no writer inside it is perturbing, nor reader inside it capturing.
    def __init__(self, module):
        self.module = module
        self.inside = set(module.modules())
        self.key = None

    def register(self):
        forward = self.module.forward
        self.module.forward = lambda *args: self.run(forward, *args)

    def run(self, forward, *args):
        if changed:  # a writer has changed a value before this module, which then sees different arguments
            return forward(*args)
        key = (args, [param._version for param in self.module.parameters()])
        if not self.busy() and same(key, self.key):
            return self.value
        value = forward(*args)
        if not perturbing:
            self.key, self.value = key, value
        return value

    def busy(self):
        hooks = perturbing | readers if capturing else perturbing
        return any(hook.site.module in self.inside for hook in hooks)

    def remove(self):
        del self.module.forward
        self.key = self.value = None

def same(a, b):  # tensors by value, tuples and lists by element
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(map(same, a, b))
    return a == b
