# SiteWeaver

SiteWeaver provides frozen models for binding-pocket, active-site, cryptic-site,
and allosteric-site prediction. Its two allosteric branches are SW-Allo-R
(residue ranking, R0 checkpoint) and SW-Allo-P (FPocket candidate ranking, G1
checkpoint). They were trained with **annotated active-site residues**. They do
not accept predicted active-site probabilities as an equivalent input. The
package contains its runtime source and selected checkpoints and does not
import the research project.

## Install

Use Python 3.10+ and a CPU or CUDA build of PyTorch:

```bash
python -m pip install -e .
```

FPocket is a separate executable required for SW-Allo-P. The frozen G1 cache
was generated with FPocket 4.2.3. Another FPocket version can produce a
different candidate list and therefore different scores. ProDy is optional
and is needed only for the separate historical PRS adapter:

```bash
python -m pip install -e '.[prs]'
```

## Input

Supply a protein PDB or PDB.GZ path. Ligands are not model inputs. The
allosteric commands additionally require a CSV listing annotated active-site
residues, one per row:

```csv
chain_id,residue_number,insertion_code
A,61,
A,62,
```

Every row must map to the supplied structure. Missing or duplicate residues
cause an error. `examples/4ZSG_active_sites.csv` illustrates the format. The
active-site predictor is a separate task and is **not** silently used as the
known-active input of SW-Allo-R or SW-Allo-P.

## PDB prediction

Pocket, active-site, and cryptic-site predictions need only the PDB:

```bash
python predict.py --pdb /path/to/protein.pdb \
  --out-dir /path/to/output --task all --device auto
```

With no active-site CSV, `--task all` reports both allosteric branches as
skipped in its metadata. To run SW-Allo-R:

```bash
python predict.py --pdb /path/to/protein.pdb \
  --active-sites /path/to/active_sites.csv \
  --out-dir /path/to/output --task allosteric_residue --device auto
```

To run SW-Allo-P, supply FPocket. The package writes a protein-only input,
runs FPocket, constructs the same 25-node-feature/22-edge-feature pocket-node
graph schema as training, and scores all candidates:

```bash
python predict.py --pdb /path/to/protein.pdb \
  --active-sites /path/to/active_sites.csv \
  --fpocket-bin /path/to/fpocket \
  --out-dir /path/to/output --task allosteric_pocket --device auto
```

`--task allosteric` runs both branches and requires both extra inputs. With
both inputs supplied, `--task all` runs pocket, active-site, cryptic-site,
SW-Allo-R, and SW-Allo-P. `--chain A B` restricts the model to selected
chains; active-site CSV identifiers must then refer to those chains.

SW-Allo-R writes raw per-residue scores to CSV and within-protein rank
percentiles (0-100) to PDB B-factors. SW-Allo-P writes one score per FPocket
candidate, a JSON record for the highest-scoring candidate, and a PDB whose
B-factor is 100 for residues mapped to that candidate within 4 A of an alpha
sphere and 0 elsewhere. The 6 A neighbor list used to build the G1 graph is
also saved separately; it is not the Figure 3 evaluation mask.
Allosteric outputs are **ranking scores, not calibrated probabilities**.
Pocket, active-site, and cryptic-site B-factors remain probabilities multiplied
by 100. Each prediction includes a metadata JSON with checkpoint filenames,
input paths, skipped tasks, and score semantics.

## Cached evaluation and optional tools

Existing R0/G1 packed caches can be evaluated without rerunning FPocket:

```bash
python rank_cached.py --cache-dir /path/to/g1_cache \
  --complex-id 4ZSG --out-dir /path/to/output --mode pocket --device cpu
python rank_cached.py --cache-dir /path/to/r0_cache \
  --complex-id 4ZSG --out-dir /path/to/output --mode residue --device cpu
```

FPocket can also be run separately with the study's local-copy invocation
policy:

```bash
python run_fpocket.py --fpocket-bin /path/to/fpocket \
  --pdb /path/to/protein.pdb --out-dir /path/to/fpocket_output
```

The optional `compute_prs.py` adapter reproduces the ProDy ANM/PRS feature
generation used in historical I01 controls. PRS is **not** an input of either
current allosteric branch. The retired I01 implementation, weights, and
release documentation are preserved by pre-switch commit
`0d2148b8f4446889f040b8aebf42304eb40b6afb`. The local archive described in
`archive/legacy_i01/README.md` is not part of the current GitHub payload.

## Training reproducibility

The existing R0/G1 training entry points accept externally supplied packed
caches and a two-column `complex_id,split` registry. Running them is not
required for inference and does not modify the bundled checkpoints:

```bash
python training/train_sw_allo_r.py --cache-dir /path/to/r0_cache \
  --split-file /path/to/split.csv --out-dir /path/to/run_r0 --device cuda
python training/train_sw_allo_p.py --cache-dir /path/to/g1_cache \
  --split-file /path/to/split.csv --out-dir /path/to/run_g1 --device cuda
```

The shared paper-set registry may list a complex twice, once for `val` and
once for `test`. The packed caches and registry are supplied separately.

The released active-site and cryptic-site weights correspond to the
`AS_PROB` and `CR_PROB` probability-only controls. To reproduce their training
from separately supplied split caches:

```bash
python -m pip install -e '.[training]'
python training/train_site_labels.py --task active \
  --cache-dir /path/to/as_prob_cache --out-dir /path/to/run_active --device cuda
python training/train_site_labels.py --task cryptic \
  --cache-dir /path/to/cr_prob_cache --out-dir /path/to/run_cryptic --device cuda
```

The active cache has embedded `train`, `val`, and `test` splits (941/103/116
structures). The cryptic cache has `train-0` to `train-3` and `test` splits;
training uses `train-0..2`, validation uses `train-3`, and test is untouched.
Both trainers default to 100 epochs, batch size 2, the original 8-layer
architecture, weighted BCE plus 0.2 Dice loss, and validation AUPRC checkpoint
selection. `--dry-run` checks a cached example without training.

The R0 model has 16 GNN layers, 14 atom-node features, and 21 edge features.
The G1 model has 16 GNN layers, 25 node features, and 22 edge features. Both
use protein heavy atoms; G1 adds one node per FPocket candidate. The frozen
G1 pocket-feature normalization statistics are in
`configs/g1_feature_stats.json`. Large training caches are distributed
separately, not through Git. See `configs/final_profiles.yaml`,
`weights/manifest.json`, and `release/artifact_manifest.json` for model,
checkpoint, and data provenance.

The software release is for research use. Preserve the upstream YuelPocket
and FPocket attributions and their applicable licenses when redistributing it.
