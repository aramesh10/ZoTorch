# Tools for writing zeroth-order estimators on PyTorch models.
#
#   sites.py      Output, Input, Weight: where in a module a hook reads or writes
#   hooks.py      WriterHook changes the value at a site inside perturb(), ReaderHook sees it inside a capture(),
#                 perturb({writer: change}) perturbs several writers together,
#                 CacheHook has a module return its output from the last clean pass that is already done
#   weight.py     ZoWeight, a weight by name with the hooks on its module
#   module.py     ZoModule, a module to rerun alone, with the weights inside it
#   estimator.py  Estimator, which registers each target's hooks around estimate, and seeds for noise
#   gradient.py   weight_gradient: a layer's weight gradient from the error at its output

from .estimator import Estimator, manual_seed, seed
from .gradient import weight_gradient
from .hooks import CacheHook, ReaderHook, WriterHook, capture, perturb
from .sites import Input, Output, Weight
from .module import ZoModule
from .weight import ZoWeight
