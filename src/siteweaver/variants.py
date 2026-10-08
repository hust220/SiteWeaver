from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    """Feature schema for the article-main predicted-active PRS cascade."""

    name: str
    description: str
    pretraining_mode: str = "off"
    use_pocket_probability: bool = False
    use_active_node_probability: bool = True
    use_active_context: bool = True
    use_prs: bool = True
    edge_mode: str = "all"
    n_layers: int = 8
    negative_mode: str = "both"
    same_weight: float = 1.0
    cross_weight: float = 0.5
    prs_feature_dim: int = 3
    prs_mode: str = "directional"
    node_input_dim_override: int | None = None
    edge_input_dim_override: int | None = None

    @property
    def node_input_dim(self) -> int:
        # The training registry starts from the 31-channel context schema.
        # I01 removes the direct pocket channel, retains the active channel,
        # and appends the three directional PRS channels: 31 - 1 + 3 = 33.
        dim = 31
        if not self.use_pocket_probability:
            dim -= 1
        if not self.use_active_node_probability:
            dim -= 1
        value = dim + self.prs_feature_dim
        return int(self.node_input_dim_override) if self.node_input_dim_override is not None else value

    @property
    def edge_input_dim(self) -> int:
        return int(self.edge_input_dim_override) if self.edge_input_dim_override is not None else 5


I01_NOYP_PRS = VariantSpec(
    name="I01_NOYP_PRS",
    description="Predicted active context plus directional PRS; no direct pocket input.",
)


def get_variant(name: str = "I01_NOYP_PRS") -> VariantSpec:
    if str(name).upper() != I01_NOYP_PRS.name:
        raise ValueError("This package exposes only the article-main I01_NOYP_PRS schema")
    return I01_NOYP_PRS
