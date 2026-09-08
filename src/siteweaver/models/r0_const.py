from __future__ import annotations

import numpy as np


# Hydrogen is intentionally excluded. The categories are based on the PDB
# element field and do not encode amino-acid identity.
ELEMENTS = ("C", "N", "O", "S", "P", "SE", "HALOGEN", "METAL", "UNKNOWN")
ELEMENT_TO_INDEX = {element: index for index, element in enumerate(ELEMENTS)}
N_ELEMENT_FEATURES = len(ELEMENTS)
N_ROLE_FEATURES = 3  # backbone, alpha carbon, terminal atom
NODE_FEATURE_DIM = N_ELEMENT_FEATURES + N_ROLE_FEATURES
POCKET_CONTEXT_DIM = 1
ACTIVE_CONTEXT_DIM = 1
MODEL_NODE_FEATURE_DIM = NODE_FEATURE_DIM + POCKET_CONTEXT_DIM + ACTIVE_CONTEXT_DIM

RBF_BASIS = 16
EDGE_FLAG_DIM = 5  # same residue, same chain, covalent, peptide, sequential-neighbor
EDGE_FEATURE_DIM = RBF_BASIS + EDGE_FLAG_DIM
CONTACT_CUTOFF = 5.0
MAX_NEIGHBORS = 48

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
HALOGENS = {"F", "CL", "BR", "I"}
METALS = {
    "LI", "NA", "K", "RB", "CS", "MG", "CA", "SR", "BA", "MN", "FE",
    "CO", "NI", "CU", "ZN", "CD", "HG", "AL", "GA", "IN", "SN",
}
COVALENT_RADII = {
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "SE": 1.20,
}


def normalize_element(value: str | None) -> str:
    token = "".join(ch for ch in str(value or "").upper().strip() if ch.isalpha())
    if token == "H":
        return "H"
    if token in {"SE"}:
        return "SE"
    if token in HALOGENS:
        return "HALOGEN"
    if token in METALS:
        return "METAL"
    if token in {"C", "N", "O", "S", "P"}:
        return token
    return "UNKNOWN"


def element_feature(element: str) -> np.ndarray:
    feature = np.zeros(N_ELEMENT_FEATURES + N_ROLE_FEATURES, dtype=np.float32)
    feature[ELEMENT_TO_INDEX.get(element, ELEMENT_TO_INDEX["UNKNOWN"])] = 1.0
    return feature
