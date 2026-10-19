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
from .active_sites import load_active_site_mask
from .const import NODE_FEATURE_DIM
from .cryptic_source import CrypticPocketModel
from .g1_features import build_g1_graph, write_protein_only_pdb
from .graph import Graph
from .io import write_bfactor_pdb
from .models.g1_model import FPocketPocketNodeRanker
from .pdb_utils import build_protein_only_graph_arrays, parse_pdb_residues
from .pocket_model import LigandFreeYuelPocket
from .models.r0_model import HeavyAtomR0KnownActive
from .r0_features import build_r0_graph
from .tools.fpocket import _resolve_binary, run as run_fpocket


WEIGHT_FILES = {
    "pocket": "ligand_free_pocket_final.ckpt",
    "active_site": "active_site_final.ckpt",
    "allosteric_residue": "allosteric_residue_rank_final.ckpt",
    "allosteric_pocket": "allosteric_pocket_rank_final.ckpt",
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
    """Self-contained inference for the bundled SiteWeaver tasks.

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
        elif name == "cryptic":
            model = CrypticPocketModel(
                hidden_nf=hidden_nf,
                n_layers=n_layers,
                dropout=0.0,
                node_input_dim=int(hparams.get("node_input_dim", 30)),
            )
        elif name == "allosteric_residue":
            model = HeavyAtomR0KnownActive(
                hidden_nf=hidden_nf,
                embedding_dim=int(hparams.get("embedding_dim", 64)),
                n_layers=int(hparams.get("n_layers", 16)),
            )
        elif name == "allosteric_pocket":
            model = FPocketPocketNodeRanker(dropout=0.0)
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
        residue_meta: list[dict],
        pocket_probability: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        arrays = build_r0_graph(
            pdb_path,
            residue_meta,
            pocket_probability.detach().cpu().numpy(),
            active_mask.detach().cpu().numpy(),
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
            "active_site_probability": active_mask,
            "residue_sample_ids": torch.zeros(len(residue_meta), dtype=torch.long, device=self.device),
            "num_residues_per_sample": torch.tensor([len(residue_meta)], device=self.device),
        }
        scores, scale = self._get_model("allosteric_residue")(batch)
        return scores, {
            "score_scale": float(scale.detach().cpu().item()),
            "active_context_source": "user-supplied annotated active-site residues",
            "uses_prs": False,
            "profile": "SW-Allo-R/R0",
        }

    @torch.inference_mode()
    def _allosteric_score_g1(
        self,
        pdb_path: Path,
        out_dir: Path,
        residue_meta: list[dict],
        pocket_probability: torch.Tensor,
        active_mask: torch.Tensor,
        fpocket_bin: str | Path,
    ) -> tuple[torch.Tensor, list[dict], dict]:
        protein_pdb = out_dir / f"{_input_stem(pdb_path)}_protein_only.pdb"
        write_protein_only_pdb(pdb_path, protein_pdb, residue_meta)
        fingerprint = hashlib.sha256()
        fingerprint.update(protein_pdb.read_bytes())
        fingerprint.update(_resolve_binary(fpocket_bin).read_bytes())
        fpocket_root = out_dir / "fpocket" / fingerprint.hexdigest()[:16]
        report = run_fpocket(fpocket_bin, protein_pdb, fpocket_root, workers=1, reuse=True)
        run = report["runs"][0]
        fpocket_output = Path(run["output_dir"]) / f"{run['complex_id']}_out"
        arrays = build_g1_graph(
            protein_pdb,
            fpocket_output,
            residue_meta,
            pocket_probability.detach().cpu().numpy(),
            active_mask.detach().cpu().numpy().astype(bool),
        )
        graph = Graph(
            torch.as_tensor(arrays["edge_index"], dtype=torch.long, device=self.device),
            int(arrays["node_features"].shape[0]),
        )
        graph.ndata["h"] = torch.as_tensor(arrays["node_features"], dtype=torch.float32, device=self.device)
        graph.ndata["pocket_node_mask"] = torch.as_tensor(arrays["pocket_node_mask"], dtype=torch.bool, device=self.device)
        graph.edata["e"] = torch.as_tensor(arrays["edge_features"], dtype=torch.float32, device=self.device)
        scores, scale = self._get_model("allosteric_pocket")({"graph": graph})
        return scores, arrays["pocket_records"], {
            "score_scale": float(scale.detach().cpu().item()),
            "active_context_source": "user-supplied annotated active-site residues",
            "uses_prs": False,
            "profile": "SW-Allo-P/G1",
            "fpocket_binary": report["binary"],
            "fpocket_output": str(fpocket_output),
            "num_candidates": len(arrays["pocket_records"]),
        }

    def _write_pocket_rank(
        self, pdb_path: Path, out_dir: Path, residue_meta: list[dict], scores: torch.Tensor, records: list[dict]
    ) -> dict:
        values = scores.detach().cpu().numpy().astype(np.float32)
        if len(values) != len(records):
            raise ValueError("G1 returned a different number of scores than FPocket candidates")
        percentiles = _rank_percentile(values)
        top_index = int(values.argmax())
        csv_path = out_dir / f"{_input_stem(pdb_path)}_allosteric_pocket_scores.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pocket_number", "score", "rank_percentile", "nearby_residue_indices_6a", "mapped_residue_indices_4a"])
            writer.writeheader()
            for record, score, percentile in zip(records, values, percentiles):
                mapped = [
                    residue for residue, distance in zip(record["nearby_residue_indices"], record["pocket_residue_distances"])
                    if float(distance) <= 4.0
                ]
                writer.writerow({
                    "pocket_number": record["pocket_number"],
                    "score": f"{float(score):.8g}",
                    "rank_percentile": f"{float(percentile):.6g}",
                    "nearby_residue_indices_6a": ";".join(map(str, record["nearby_residue_indices"])),
                    "mapped_residue_indices_4a": ";".join(map(str, mapped)),
                })
        selected = records[top_index]
        mapped = [
            residue for residue, distance in zip(selected["nearby_residue_indices"], selected["pocket_residue_distances"])
            if float(distance) <= 4.0
        ]
        selected_path = out_dir / f"{_input_stem(pdb_path)}_allosteric_pocket_top1.json"
        selected_path.write_text(json.dumps({
            "pocket_number": selected["pocket_number"],
            "score": float(values[top_index]),
            "nearby_residue_indices_6a": selected["nearby_residue_indices"],
            "mapped_residue_indices_4a": mapped,
            "selection_rule": "highest SW-Allo-P raw ranking score",
        }, indent=2, sort_keys=True) + "\n")
        bfactor = np.zeros(len(residue_meta), dtype=np.float32)
        bfactor[mapped] = 100.0
        pdb_out = out_dir / f"{_input_stem(pdb_path)}_allosteric_pocket_top1.pdb"
        write_bfactor_pdb(pdb_path, pdb_out, bfactor.tolist(), residue_meta)
        return {
            "csv": str(csv_path), "selected_pocket": str(selected_path), "pdb": str(pdb_out),
            "score_type": "ranking_score", "pdb_bfactor_semantics": "top1_pocket_residue_mask_4a_percent",
            "num_candidates": len(records),
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

    def predict(
        self,
        pdb_path: str | Path,
        out_dir: str | Path,
        task: str = "all",
        active_sites: str | Path | None = None,
        fpocket_bin: str | Path | None = None,
    ) -> dict:
        pdb_path = Path(pdb_path).expanduser().resolve()
        out_dir = Path(out_dir).expanduser().resolve()
        if not pdb_path.is_file():
            raise FileNotFoundError(pdb_path)
        task = str(task).lower()
        aliases = {"active": "active_site", "allosteric_site": "allosteric", "allosteric_r0": "allosteric_residue", "cryptic_site": "cryptic"}
        task = aliases.get(task, task)
        valid = {"all", "pocket", "active_site", "allosteric", "allosteric_residue", "allosteric_pocket", "cryptic"}
        if task not in valid:
            raise ValueError(f"task must be one of {sorted(valid)}")
        if task.startswith("allosteric") and active_sites is None:
            raise ValueError("R0/G1 require --active-sites with annotated residues; predicted active-site probabilities are not interchangeable")
        if task in {"allosteric", "allosteric_pocket"} and fpocket_bin is None:
            raise ValueError("SW-Allo-P requires --fpocket-bin")
        out_dir.mkdir(parents=True, exist_ok=True)

        graph, residue_meta, base_features = self._graph(pdb_path)
        n_residues = len(residue_meta)
        results = {}
        skipped = {}
        active_mask = None
        if active_sites is not None and task in {"all", "allosteric", "allosteric_residue", "allosteric_pocket"}:
            active_mask = torch.as_tensor(
                load_active_site_mask(active_sites, residue_meta), dtype=torch.float32, device=self.device
            )
        pocket_probability = None
        if task in {"all", "pocket", "active_site", "cryptic", "allosteric", "allosteric_residue", "allosteric_pocket"}:
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
        if task in {"all", "allosteric", "allosteric_residue"} and active_mask is not None:
            allosteric_score, allosteric_stats = self._allosteric_score_r0(
                pdb_path, residue_meta, pocket_probability, active_mask
            )
            results["allosteric_residue"] = self._write_task(
                "allosteric_residue", pdb_path, out_dir, residue_meta, allosteric_score,
                "ranking_score", _rank_percentile(allosteric_score.detach().cpu().numpy()),
            )
            results["allosteric_residue"].update(allosteric_stats)
        elif task == "all":
            skipped["allosteric_residue"] = "annotated active-site CSV was not provided"
        if task in {"all", "allosteric", "allosteric_pocket"} and active_mask is not None and fpocket_bin is not None:
            pocket_scores, records, pocket_stats = self._allosteric_score_g1(
                pdb_path, out_dir, residue_meta, pocket_probability, active_mask, fpocket_bin
            )
            results["allosteric_pocket"] = self._write_pocket_rank(
                pdb_path, out_dir, residue_meta, pocket_scores, records
            )
            results["allosteric_pocket"].update(pocket_stats)
        elif task == "all":
            skipped["allosteric_pocket"] = (
                "annotated active-site CSV was not provided" if active_mask is None else "FPocket binary was not provided"
            )
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
            "version": "0.2.0",
            "input_pdb": str(pdb_path),
            "chains": list(self.chains) or "all",
            "device": str(self.device),
            "num_residues": n_residues,
            "num_graph_nodes": graph.num_nodes,
            "num_graph_edges": graph.num_edges,
            "task": task,
            "checkpoints": dict(self.checkpoints),
            "annotated_active_sites": str(Path(active_sites).expanduser().resolve()) if active_mask is not None else None,
            "skipped_tasks": skipped,
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
