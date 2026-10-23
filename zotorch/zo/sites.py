# Sites: where in a module a hook reads or writes. read(see) calls see(get) on each forward pass, where get() returns
# the value, and write(change) has the module use change(value) instead, or value as it is when change returns None.
# Both return the hooks' handles.

class Output:  # a module's output
    def __init__(self, module):
        self.module = module

    def read(self, see):  # cloned, since the model may change its outputs in place
        return [self.module.register_forward_hook(lambda m, args, out: see(lambda: out.clone()))]

    def write(self, change):
        return [self.module.register_forward_hook(lambda m, args, out: change(out))]

class Input:  # one of a module's positional arguments
    def __init__(self, module, index):
        self.module, self.index = module, index

    def read(self, see):
        return [self.module.register_forward_pre_hook(lambda m, args: see(lambda: args[self.index]))]

    def write(self, change):
        i = self.index

        def hook(m, args):
            value = change(args[i])
            return None if value is None else (*args[:i], value, *args[i + 1:])
        return [self.module.register_forward_pre_hook(hook)]

class Weight:  # parameters the module's forward uses; change(params) changes them in place and returns its undo
    def __init__(self, module, params):
        self.module, self.params = module, list(params)

    def read(self, see):
        return [self.module.register_forward_pre_hook(lambda m, args: see(lambda: self.params))]

    def write(self, change):
        undo = []

        def before(m, args):
            undo.append(change(self.params))

        def after(m, args, out):
            restore = undo.pop()
            if restore is not None:
                restore()
        return [self.module.register_forward_pre_hook(before), self.module.register_forward_hook(after)]
