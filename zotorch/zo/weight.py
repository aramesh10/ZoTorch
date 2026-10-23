from .hooks import ReaderHook, WriterHook
from .sites import Input, Output, Weight

class ZoWeight:
    """Wrapper around weight parameter.

        weight = ZoWeight(model, 'transformer.h.0.mlp.c_fc.weight')

    Attributes:
        param         the weight tensor in the model
        module        the module that owns it
        input         reader of the module's first argument
        output        reader of the module's output
        write_input   writer of the module's first argument
        write_output  writer of the module's output
        write_weight  writer that changes the weight during the module's forward pass, then undoes the change
        hooks         all five, readers first

    output: where the module's output is seen. By default its own output; for a layer whose output enters another
    module as an argument, that argument, e.g. Input(attn, 1).
    """

    def __init__(self, model, name, output=None):
        path, _, attr = name.rpartition('.')
        self.name = name
        self.module = model.get_submodule(path)
        self.param = getattr(self.module, attr)

        output = Output(self.module) if output is None else output

        self.input = ReaderHook(Input(self.module, 0))
        self.output = ReaderHook(output)
        self.write_input = WriterHook(Input(self.module, 0))
        self.write_output = WriterHook(output)
        self.write_weight = WriterHook(Weight(self.module, [self.param]))

        self.hooks = [self.input, self.output, self.write_input, self.write_output, self.write_weight]
