from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    """The fixed allosteric configuration bundled with this release."""

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
    cross_weight: float = 1.0
    prs_feature_dim: int = 3
    prs_mode: str = "directional"

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
        return dim + self.prs_feature_dim


I01_NOYP_PRS = VariantSpec(
    name="I01_NOYP_PRS",
    description="Predicted active context plus directional PRS; no direct pocket input.",
)


def get_variant(name: str = "I01_NOYP_PRS") -> VariantSpec:
    if str(name).upper() != I01_NOYP_PRS.name:
        raise ValueError("This package bundles only the selected I01_NOYP_PRS allosteric model")
    return I01_NOYP_PRS
