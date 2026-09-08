import torch


TORCH_FLOAT = torch.float32
TORCH_INT = torch.int32

ALLOWED_RESIDUE_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL", "SEP", "TPO", "PTR",
    "UNK", "BB",
]

RESIDUE2IDX = {res: idx for idx, res in enumerate(ALLOWED_RESIDUE_TYPES)}
IDX2RESIDUE = {idx: res for res, idx in RESIDUE2IDX.items()}
N_RESIDUE_TYPES = len(RESIDUE2IDX)

# residue one-hot + node flags: is_bb, is_sc
NODE_FEATURE_DIM = N_RESIDUE_TYPES + 2

# dist_norm, contact, bb_sc_same_residue, sequential, same_chain
EDGE_FEATURE_DIM = 5


def residue_name_to_index(name: str) -> int:
    return RESIDUE2IDX.get(str(name).upper(), RESIDUE2IDX["UNK"])

