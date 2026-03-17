"""Model entrypoint for A09."""

from dyt import DynamicTanh
from replace_norm import replace_layernorm_with_dyt

__all__ = ["DynamicTanh", "replace_layernorm_with_dyt"]
