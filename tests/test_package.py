import json
from pathlib import Path

import pytest
import torch

from siteweaver.inference import SiteWeaverPredictor
from siteweaver.models.g1_model import FPocketPocketNodeRanker
from siteweaver.models.r0_model import HeavyAtomR0KnownActive
from siteweaver.active_sites import load_active_site_mask
from siteweaver.rank_loss import allosteric_rank_loss
from siteweaver.models.r0_variants import R0


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
    for name in ("pocket", "active_site", "allosteric_residue", "allosteric_pocket", "cryptic"):
        assert predictor._get_model(name).training is False


def test_final_rankers_load():
    pocket = FPocketPocketNodeRanker()
    pocket.load_state_dict(_state(WEIGHTS / "allosteric_pocket_rank_final.ckpt"), strict=True)
    residue = HeavyAtomR0KnownActive()
    residue.load_state_dict(_state(WEIGHTS / "allosteric_residue_rank_final.ckpt"), strict=True)


def test_active_site_example_maps():
    from siteweaver.pdb_utils import parse_pdb_residues

    residues = parse_pdb_residues(ROOT / "examples/4ZSG_input.pdb")
    mask = load_active_site_mask(ROOT / "examples/4ZSG_active_sites.csv", residues)
    assert int(mask.sum()) == 22


def test_allosteric_requires_annotated_active_sites(tmp_path):
    predictor = SiteWeaverPredictor(device="cpu")
    with pytest.raises(ValueError, match="require --active-sites"):
        predictor.predict(ROOT / "examples/4ZSG_input.pdb", tmp_path, task="allosteric_residue")


def test_r0_rank_loss_retains_gradient():
    scores = torch.tensor([2.0, 0.0, 1.0, -1.0], requires_grad=True)
    loss, metrics = allosteric_rank_loss(
        scores,
        torch.tensor(1.0),
        torch.tensor([False, False, False, False]),
        torch.tensor([True, False, True, False]),
        torch.tensor([0, 0, 1, 1]),
        R0,
        negatives_per_positive=2,
        cross_protein_negatives_per_positive=2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(scores.grad).all()
    assert metrics["num_positive_residues"].item() == 2


def test_g1_visualization_uses_four_angstrom_mapping(tmp_path):
    from siteweaver.pdb_utils import parse_pdb_residues

    pdb = ROOT / "examples/4ZSG_input.pdb"
    residues = parse_pdb_residues(pdb)
    records = [
        {"pocket_number": 1, "nearby_residue_indices": [0, 1], "pocket_residue_distances": [3.5, 5.0]},
        {"pocket_number": 2, "nearby_residue_indices": [2], "pocket_residue_distances": [3.0]},
    ]
    result = SiteWeaverPredictor(device="cpu")._write_pocket_rank(
        pdb, tmp_path, residues, torch.tensor([2.0, 1.0]), records
    )
    selected = json.loads(Path(result["selected_pocket"]).read_text())
    assert selected["nearby_residue_indices_6a"] == [0, 1]
    assert selected["mapped_residue_indices_4a"] == [0]
    assert result["pdb_bfactor_semantics"] == "top1_pocket_residue_mask_4a_percent"


def test_training_registry_allows_shared_validation_and_test(tmp_path):
    from training._common import split_samples

    registry = tmp_path / "splits.csv"
    registry.write_text("complex_id,split\nA,train\nB,val\nB,test\n")
    samples = [{"complex_id": "A"}, {"complex_id": "B"}]
    assert [item["complex_id"] for item in split_samples(samples, registry, "train")] == ["A"]
    assert [item["complex_id"] for item in split_samples(samples, registry, "val")] == ["B"]
    assert [item["complex_id"] for item in split_samples(samples, registry, "test")] == ["B"]
