import inspect

from .hooks import ReaderHook
from .sites import Input
from .weight import ZoWeight

class ZoModule:
    """Wrapper around nn.Module.

        module = ZoModule(model, 'transformer.h.1.attn', read='transformer.h.1.attn.c_proj.weight',
                          layers={'c_q': 'transformer.h.1.attn.c_q.weight', ...},
                          arguments={'ve': (1, 'value_embeds.1.weight')})

    Arguments:
        name       the module to rerun
        read       the weight of the layer whose input replay() returns
        layers     {kind: weight name}: layers inside the module, seen at their outputs
        arguments  {kind: (index, weight name)}: layers outside it, seen as the module's argument at index

    Attributes:
        module     the module
        args       a reader for each of its positional arguments
        read       the ZoWeight of the read layer
        weights    {kind: ZoWeight} for layers and arguments
        hooks      the readers, every weight's hooks, and the ZoModule itself
    """

    def __init__(self, model, name, read, layers, arguments):
        self.module = model.get_submodule(name)
        count = len(inspect.signature(self.module.forward).parameters)
        self.args = [ReaderHook(Input(self.module, i)) for i in range(count)]
        self.read = ZoWeight(model, read)
        self.weights  = {kind: ZoWeight(model, weight) for kind, weight in layers.items()}
        self.weights |= {kind: ZoWeight(model, weight, Input(self.module, i))
                         for kind, (i, weight) in arguments.items()}
        weights = [self.read, *self.weights.values()]
        self.hooks = [*self.args, *(hook for weight in weights for hook in weight.hooks), self]

    def replay(self):
        """Run the module alone on its arguments from the last capture; return the value at read's input."""
        self.module(*(reader.value for reader in self.args))
        return self.value

    def register(self):
        """Record read's input on every pass, inside a capture or not, for replay() to return."""
        self.handles = Input(self.read.module, 0).read(self.see)

    def see(self, get):
        self.value = get()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.value = None
