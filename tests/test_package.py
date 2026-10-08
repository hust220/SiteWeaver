from pathlib import Path

import torch

from siteweaver.inference import SiteWeaverPredictor
from siteweaver.models.g1_model import FPocketPocketNodeRanker
from siteweaver.models.r0_model import HeavyAtomR0KnownActive
from siteweaver.ranker_model import SiteWeaverAblationModel
from siteweaver.variants import get_variant


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"


def _state(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    return {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state.items()
    }


def test_runtime_profiles_load():
    predictor = SiteWeaverPredictor(device="cpu")
    for name in ("pocket", "active_site", "active_context", "allosteric", "allosteric_r0", "cryptic"):
        assert predictor._get_model(name).training is False


def test_final_rankers_load():
    pocket = FPocketPocketNodeRanker()
    pocket.load_state_dict(_state(WEIGHTS / "allosteric_pocket_rank_final.ckpt"), strict=True)
    residue = HeavyAtomR0KnownActive()
    residue.load_state_dict(_state(WEIGHTS / "allosteric_residue_rank_final.ckpt"), strict=True)
    i01 = SiteWeaverAblationModel(get_variant("I01_NOYP_PRS"))
    i01.load_state_dict(_state(WEIGHTS / "allosteric_i01_prs_final.ckpt"), strict=True)
