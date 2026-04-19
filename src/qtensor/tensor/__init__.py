# src/xtensor/__init__.py

import jax

jax.config.update("jax_enable_x64", True)

from .index import Index
from .tensor import Tensor
from .contract import contract