from __future__ import annotations

from . import r0_const


BASE_NODE_FEATURE_DIM = r0_const.MODEL_NODE_FEATURE_DIM
BASE_EDGE_FEATURE_DIM = r0_const.EDGE_FEATURE_DIM
POCKET_NODE_EXTRA_DIM = 11
NODE_FEATURE_DIM = BASE_NODE_FEATURE_DIM + POCKET_NODE_EXTRA_DIM
EDGE_FEATURE_DIM = BASE_EDGE_FEATURE_DIM + 1
POCKET_LINK_FLAG_INDEX = BASE_EDGE_FEATURE_DIM
POCKET_NODE_FLAG_INDEX = BASE_NODE_FEATURE_DIM
POCKET_FEATURE_NAMES = (
    "is_pocket_node",
    "log1p_alpha_sphere_count",
    "log1p_nearby_residue_count",
    "mean_alpha_sphere_radius",
    "log1p_volume",
    "mean_alpha_sphere_solvent_accessibility",
    "mean_ligand_free_pocket_probability",
    "max_ligand_free_pocket_probability",
    "active_residue_fraction",
    "min_active_distance",
    "mean_active_distance",
)
STANDARDIZED_POCKET_FEATURES = {
    "log1p_alpha_sphere_count",
    "log1p_nearby_residue_count",
    "mean_alpha_sphere_radius",
    "log1p_volume",
    "mean_alpha_sphere_solvent_accessibility",
    "min_active_distance",
    "mean_active_distance",
}
POCKET_NEIGHBOR_DISTANCE = 6.0
POCKET_EDGE_RBF_CUTOFF = POCKET_NEIGHBOR_DISTANCE
RBF_BASIS = r0_const.RBF_BASIS
HIDDEN_NF = 64
EMBEDDING_DIM = 64
N_LAYERS = 16
MARGIN = 0.2
SAME_PROTEIN_LOSS_WEIGHT = 1.0
CROSS_PROTEIN_LOSS_WEIGHT = 0.5
NEGATIVES_PER_POSITIVE = 32
CROSS_PROTEIN_NEGATIVES_PER_POSITIVE = 16
