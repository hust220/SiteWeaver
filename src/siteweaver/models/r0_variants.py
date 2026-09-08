from __future__ import annotations

from dataclasses import dataclass

from . import r0_const as config


@dataclass(frozen=True)
class VariantSpec:
    name: str
    n_layers: int = 16
    negative_mode: str = "both"
    same_weight: float = 1.0
    cross_weight: float = 0.5

    @property
    def node_input_dim(self) -> int:
        return config.MODEL_NODE_FEATURE_DIM


R0 = VariantSpec(name="R0")


def get_variant(name: str) -> VariantSpec:
    if str(name).upper() != "R0":
        raise ValueError(f"This module contains only the frozen R0 configuration, got {name!r}")
    return R0
