from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .active_source import SiteWeaverActiveSiteModel
from .const import NODE_FEATURE_DIM
from .cryptic_source import CrypticPocketModel
from .graph import Graph
from .io import write_bfactor_pdb
from .pdb_utils import build_protein_only_graph_arrays, parse_pdb_residues
from .pocket_model import LigandFreeYuelPocket
from .prs_features import build_prs_features
from .models.r0_model import HeavyAtomR0KnownActive
from .ranker_model import SiteWeaverAblationModel
from .variants import get_variant
from .r0_features import build_r0_graph


WEIGHT_FILES = {
    "pocket": "ligand_free_pocket_final.ckpt",
    "active_site": "active_site_final.ckpt",
    "active_context": "active_context_final.ckpt",
    "allosteric": "allosteric_i01_prs_final.ckpt",
    "allosteric_r0": "allosteric_residue_rank_final.ckpt",
    "cryptic": "cryptic_site_final.ckpt",
}


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state_dict.items()
    }
    checkpoint["_runtime_state_dict"] = state_dict
    return checkpoint


def _load_model(model: torch.nn.Module, checkpoint: dict, path: Path) -> torch.nn.Module:
    model.load_state_dict(checkpoint["_runtime_state_dict"], strict=True)
    model.eval()
    model._siteweaver_checkpoint = str(path)  # type: ignore[attr-defined]
    return model


def _aggregate(values: torch.Tensor, residue_index: torch.Tensor, n_residues: int) -> torch.Tensor:
    result = values.new_zeros((int(n_residues), values.shape[-1]))
    counts = values.new_zeros((int(n_residues), 1))
    result.index_add_(0, residue_index.long(), values)
    counts.index_add_(
        0,
        residue_index.long(),
        torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device),
    )
    return result / counts.clamp_min(1.0)


def _input_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".pdb.gz"):
        return name[:-7]
    return path.stem


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.full(values.shape, 100.0, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(order.size, dtype=np.float32)
    ranks[order] = np.arange(order.size, dtype=np.float32)
    return (100.0 * ranks / float(order.size - 1)).astype(np.float32)


def _default_weights_dir() -> Path:
    candidates = (
        Path(__file__).resolve().parents[2] / "weights",
        Path(sys.prefix) / "siteweaver" / "weights",
        Path(__file__).resolve().parents[4] / "siteweaver" / "weights",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


class SiteWeaverPredictor:
    """Self-contained inference for the four bundled SiteWeaver tasks.

    The public model outputs are residue-level. Pocket, active-site, and
    cryptic outputs are probabilities. Allosteric output is the trained
    ranking score; its percentile is used for visualization and B-factors.
    """

    def __init__(
        self,
        weights_dir: str | Path | None = None,
        device: str = "auto",
        chains: Iterable[str] | None = None,
    ):
        self.package_dir = Path(__file__).resolve().parents[2]
        self.weights_dir = Path(weights_dir) if weights_dir else _default_weights_dir()
        self.device = self._resolve_device(device)
        self.chains = tuple(str(chain).strip() for chain in (chains or ()) if str(chain).strip())
        self.models: dict[str, torch.nn.Module] = {}
        self.checkpoints: dict[str, str] = {}

    @staticmethod
    def _resolve_device(value: str) -> torch.device:
        value = str(value).lower()
        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        if value.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device(value)

    def _checkpoint(self, name: str) -> tuple[dict, Path]:
        if name not in WEIGHT_FILES:
            raise KeyError(name)
        path = self.weights_dir / WEIGHT_FILES[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled checkpoint: {path}")
        checkpoint = _load_checkpoint(path, self.device)
        self.checkpoints[name] = path.name
        return checkpoint, path

    def _get_model(self, name: str) -> torch.nn.Module:
        if name in self.models:
            return self.models[name]
        checkpoint, path = self._checkpoint(name)
        hparams = checkpoint.get("hyper_parameters", {})
        hidden_nf = int(hparams.get("hidden_nf", 64))
        n_layers = int(hparams.get("n_layers", 8))
        if name == "pocket":
            model = LigandFreeYuelPocket(
                hidden_nf=hidden_nf, n_layers=n_layers, dropout=0.0, gnn_backend="reference"
            )
        elif name == "active_site":
            model = SiteWeaverActiveSiteModel(
                hidden_nf=hidden_nf,
                n_layers=n_layers,
                dropout=0.0,
                node_input_dim=int(hparams.get("node_input_dim", 30)),
            )
        elif name == "active_context":
            model = SiteWeaverActiveSiteModel(
                hidden_nf=hidden_nf,
                n_layers=n_layers,
                dropout=0.0,
                node_input_dim=int(hparams.get("node_input_dim", 29)),
            )
        elif name == "cryptic":
            model = CrypticPocketModel(
                hidden_nf=hidden_nf,
                n_layers=n_layers,
                dropout=0.0,
                node_input_dim=int(hparams.get("node_input_dim", 30)),
            )
        elif name == "allosteric":
            model = SiteWeaverAblationModel(
                spec=get_variant("I01_NOYP_PRS"),
                hidden_nf=hidden_nf,
                embedding_dim=int(hparams.get("embedding_dim", 64)),
                dropout=0.0,
            )
        elif name == "allosteric_r0":
            model = HeavyAtomR0KnownActive(
                hidden_nf=hidden_nf,
                embedding_dim=int(hparams.get("embedding_dim", 64)),
                n_layers=int(hparams.get("n_layers", 16)),
            )
        else:
            raise ValueError(f"Unknown model profile: {name}")
        model = _load_model(model.to(self.device), checkpoint, path)
        self.models[name] = model
        return model

    def _graph(self, pdb_path: Path) -> tuple[Graph, list[dict], np.ndarray]:
        residues = parse_pdb_residues(pdb_path)
        if self.chains:
            allowed = set(self.chains)
            residues = [item for item in residues if item.get("chain_id", "") in allowed]
        if not residues:
            chain_text = ", ".join(self.chains) if self.chains else "all chains"
            raise ValueError(f"No protein residues found for {chain_text} in {pdb_path}")
        arrays = build_protein_only_graph_arrays(residues, contact_cutoff=8.0)
        if arrays is None:
            raise ValueError(f"Could not build a graph from {pdb_path}")
        edge_index = torch.as_tensor(arrays["edge_index"], dtype=torch.long, device=self.device)
        graph = Graph(edge_index, int(arrays["node_features"].shape[0]))
        graph.ndata["h"] = torch.as_tensor(
            arrays["node_features"], dtype=torch.float32, device=self.device
        )
        residue_index = torch.as_tensor(
            arrays["residue_indices"], dtype=torch.long, device=self.device
        )
        graph.ndata["residue_index"] = residue_index
        graph.ndata["global_residue_index"] = residue_index
        graph.edata["e"] = torch.as_tensor(
            arrays["edge_features"], dtype=torch.float32, device=self.device
        )
        graph.meta = {"pdb_path": str(pdb_path)}
        return graph, arrays["residue_meta"], graph.ndata["h"].clone()

    @torch.inference_mode()
    def _pocket_probability(self, graph: Graph, n_residues: int) -> torch.Tensor:
        model = self._get_model("pocket")
        node_logits = model(graph).reshape(-1, 1)
        residue_logits = _aggregate(node_logits, graph.ndata["residue_index"], n_residues).squeeze(-1)
        return torch.sigmoid(residue_logits)

    @torch.inference_mode()
    def _site_probability(
        self,
        model_name: str,
        graph: Graph,
        base_features: torch.Tensor,
        n_residues: int,
        extra_residue_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if extra_residue_features is not None:
            node_features = torch.cat(
                [base_features, extra_residue_features[graph.ndata["residue_index"]]], dim=-1
            )
        else:
            node_features = base_features
        graph.ndata["h"] = node_features
        model = self._get_model(model_name)
        batch = {
            "graph": graph,
            "num_residues_per_sample": torch.tensor([n_residues], device=self.device),
        }
        logits = model(batch)
        graph.ndata["h"] = base_features
        return torch.sigmoid(logits)

    @torch.inference_mode()
    def _allosteric_score_r0(
        self,
        pdb_path: Path,
        graph: Graph,
        residue_meta: list[dict],
        pocket_probability: torch.Tensor,
        active_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        arrays = build_r0_graph(
            pdb_path,
            residue_meta,
            pocket_probability.detach().cpu().numpy(),
            active_probability.detach().cpu().numpy(),
        )
        edge_index = torch.as_tensor(arrays["edge_index"], dtype=torch.long, device=self.device)
        graph = Graph(edge_index, int(arrays["node_features"].shape[0]))
        graph.ndata["h"] = torch.as_tensor(
            arrays["node_features"], dtype=torch.float32, device=self.device
        )
        graph.ndata["global_residue_index"] = torch.as_tensor(
            arrays["residue_indices"], dtype=torch.long, device=self.device
        )
        graph.edata["e"] = torch.as_tensor(
            arrays["edge_features"], dtype=torch.float32, device=self.device
        )
        batch = {
            "graph": graph,
            "active_site_probability": active_probability,
            "residue_sample_ids": torch.zeros(len(residue_meta), dtype=torch.long, device=self.device),
            "num_residues_per_sample": torch.tensor([len(residue_meta)], device=self.device),
        }
        scores, scale = self._get_model("allosteric_r0")(batch)
        return scores, {
            "score_scale": float(scale.detach().cpu().item()),
            "active_context_source": "bundled active-context model",
            "uses_prs": False,
            "profile": "R0",
        }

    @torch.inference_mode()
    def _allosteric_score_i01(
        self,
        pdb_path: Path,
        graph: Graph,
        residue_meta: list[dict],
        active_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Run the article's predicted-active I01_NOYP_PRS cascade.

        The production ranker consumes the 29 base residue-graph channels,
        one continuous predicted active-site channel, and three CA-only PRS
        channels. It intentionally does not consume the ligand-free pocket
        probability or the ligand-free hidden embedding.
        """
        prs_features, prs_stats = build_prs_features(
            pdb_path,
            residue_meta,
            active_probability.detach().cpu().numpy(),
            cutoff=15.0,
            gamma=1.0,
            n_modes=20,
            dense_max_residues=1200,
            chain_ids=self.chains or None,
        )
        residue_index = graph.ndata["residue_index"].long()
        base_features = graph.ndata["h"]
        active_node = active_probability[residue_index, None]
        prs_node = torch.as_tensor(prs_features, dtype=base_features.dtype, device=self.device)[residue_index]
        graph_i01 = graph.clone()
        graph_i01.ndata["h"] = torch.cat([base_features, active_node, prs_node], dim=-1)
        graph_i01.ndata["global_residue_index"] = residue_index
        batch = {
            "graph": graph_i01,
            "active_site_probability": active_probability,
            "residue_sample_ids": torch.zeros(len(residue_meta), dtype=torch.long, device=self.device),
            "num_residues_per_sample": torch.tensor([len(residue_meta)], device=self.device),
        }
        scores, scale = self._get_model("allosteric")(batch)
        return scores, {
            "score_scale": float(scale.detach().cpu().item()),
            "active_context_source": "bundled base-only active-context model",
            "uses_prs": True,
            "profile": "I01_NOYP_PRS",
            "prs": prs_stats,
        }

    def _write_task(
        self,
        task: str,
        pdb_path: Path,
        out_dir: Path,
        residue_meta: list[dict],
        scores: torch.Tensor,
        score_type: str,
        bfactor_values: np.ndarray,
    ) -> dict:
        score_array = scores.detach().cpu().numpy().astype(np.float32)
        probability = score_array if score_type == "probability" else None
        percentiles = _rank_percentile(score_array)
        csv_path = out_dir / f"{_input_stem(pdb_path)}_{task}_scores.csv"
        pdb_out = out_dir / f"{_input_stem(pdb_path)}_{task}.pdb"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "residue_index",
                    "chain_id",
                    "residue_number",
                    "insertion_code",
                    "residue_name",
                    "score",
                    "score_type",
                    "probability",
                    "rank_percentile",
                ],
            )
            writer.writeheader()
            for idx, meta in enumerate(residue_meta):
                writer.writerow(
                    {
                        "residue_index": idx,
                        "chain_id": meta["chain_id"],
                        "residue_number": meta["res_num"],
                        "insertion_code": meta.get("ins_code", ""),
                        "residue_name": meta["res_name"],
                        "score": f"{float(score_array[idx]):.8g}",
                        "score_type": score_type,
                        "probability": "" if probability is None else f"{float(probability[idx]):.8g}",
                        "rank_percentile": f"{float(percentiles[idx]):.6g}",
                    }
                )
        pdb_info = write_bfactor_pdb(pdb_path, pdb_out, bfactor_values.tolist(), residue_meta)
        return {
            "csv": str(csv_path),
            "pdb": str(pdb_out),
            "score_type": score_type,
            "pdb_bfactor_semantics": "probability_percent" if score_type == "probability" else "rank_percentile",
            "pdb_mapping": pdb_info,
            "score_min": float(score_array.min()),
            "score_max": float(score_array.max()),
        }

    def predict(self, pdb_path: str | Path, out_dir: str | Path, task: str = "all") -> dict:
        pdb_path = Path(pdb_path).expanduser().resolve()
        out_dir = Path(out_dir).expanduser().resolve()
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)
        task = str(task).lower()
        aliases = {"active": "active_site", "allosteric_site": "allosteric", "cryptic_site": "cryptic"}
        task = aliases.get(task, task)
        valid = {"all", "pocket", "active_site", "active_context", "allosteric", "allosteric_r0", "cryptic"}
        if task not in valid:
            raise ValueError(f"task must be one of {sorted(valid)}")
        out_dir.mkdir(parents=True, exist_ok=True)

        graph, residue_meta, base_features = self._graph(pdb_path)
        n_residues = len(residue_meta)
        results = {}
        pocket_probability = None
        if task in {"all", "pocket", "active_site", "cryptic", "allosteric_r0"}:
            pocket_probability = self._pocket_probability(graph, n_residues)
        if task in {"all", "pocket"}:
            results["pocket"] = self._write_task(
                "pocket", pdb_path, out_dir, residue_meta, pocket_probability,
                "probability", (pocket_probability.detach().cpu().numpy() * 100.0),
            )
        if task in {"all", "active_site"}:
            active_probability = self._site_probability(
                "active_site", graph, base_features, n_residues, pocket_probability[:, None]
            )
            results["active_site"] = self._write_task(
                "active_site", pdb_path, out_dir, residue_meta, active_probability,
                "probability", (active_probability.detach().cpu().numpy() * 100.0),
            )
        else:
            active_probability = None
        active_context_probability = None
        if task in {"all", "active_context"}:
            active_context_probability = self._site_probability(
                "active_context", graph, base_features, n_residues
            )
            if task == "active_context":
                results["active_context"] = self._write_task(
                    "active_context", pdb_path, out_dir, residue_meta, active_context_probability,
                    "probability", (active_context_probability.detach().cpu().numpy() * 100.0),
                )
        if task in {"all", "allosteric"}:
            if active_context_probability is None:
                active_context_probability = self._site_probability(
                    "active_context", graph, base_features, n_residues
                )
            allosteric_score, allosteric_stats = self._allosteric_score_i01(
                pdb_path, graph, residue_meta, active_context_probability
            )
            results["allosteric"] = self._write_task(
                "allosteric", pdb_path, out_dir, residue_meta, allosteric_score,
                "ranking_score", _rank_percentile(allosteric_score.detach().cpu().numpy()),
            )
            results["allosteric"].update(allosteric_stats)
        if task == "allosteric_r0":
            if active_context_probability is None:
                active_context_probability = self._site_probability(
                    "active_context", graph, base_features, n_residues
                )
            allosteric_score, allosteric_stats = self._allosteric_score_r0(
                pdb_path, graph, residue_meta, pocket_probability, active_context_probability
            )
            results["allosteric_r0"] = self._write_task(
                "allosteric_r0", pdb_path, out_dir, residue_meta, allosteric_score,
                "ranking_score", _rank_percentile(allosteric_score.detach().cpu().numpy()),
            )
            results["allosteric_r0"].update(allosteric_stats)
        if task in {"all", "cryptic"}:
            cryptic_probability = self._site_probability(
                "cryptic", graph, base_features, n_residues, pocket_probability[:, None]
            )
            results["cryptic"] = self._write_task(
                "cryptic", pdb_path, out_dir, residue_meta, cryptic_probability,
                "probability", (cryptic_probability.detach().cpu().numpy() * 100.0),
            )

        metadata = {
            "package": "SiteWeaver",
            "version": "0.1.0",
            "input_pdb": str(pdb_path),
            "chains": list(self.chains) or "all",
            "device": str(self.device),
            "num_residues": n_residues,
            "num_graph_nodes": graph.num_nodes,
            "num_graph_edges": graph.num_edges,
            "task": task,
            "checkpoints": dict(self.checkpoints),
            "results": results,
        }
        metadata_path = out_dir / f"{_input_stem(pdb_path)}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        metadata["metadata_json"] = str(metadata_path)
        return metadata


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
