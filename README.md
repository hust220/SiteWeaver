# SiteWeaver

SiteWeaver is a self-contained inference package for four protein-site tasks:
binding-pocket, active/functional-site, allosteric-site, and cryptic-site
prediction. It includes the model source, frozen checkpoints, configuration
metadata, and one small example structure. It does not import anything from
the research project tree.

## Install

Use an environment with a CPU or CUDA build of PyTorch, then run:

```bash
python -m pip install -e .
```

The same dependencies are listed in `requirements-runtime.txt`. No
`torch_geometric` installation is required.

## PDB inference

The PDB command runs the residue-level pocket, active-site, allosteric-residue,
and cryptic-site paths. All input and output paths are supplied by the user;
the package does not assume a dataset location.

```bash
siteweaver \
  --pdb /path/to/protein.pdb \
  --out-dir /path/to/results \
  --task all \
  --device auto
```

Use `--task pocket`, `--task active_site`, `--task allosteric`, or
`--task cryptic` for one task. Use `--chain A B` to restrict the graph to
selected chains. The allosteric PDB path is the optional PRS-enabled residue
ranker; its raw score is a ranking score, not a calibrated probability.

Each task writes a CSV and a PDB with residue-level scores in the B-factor
column. Pocket, active-site, and cryptic B-factors are probabilities multiplied
by 100. Allosteric B-factors are within-protein rank percentiles from 0 to 100.
The metadata JSON records the checkpoint and score semantics.

## Production allosteric pocket ranking

The production allosteric workflow ranks FPocket candidates with one pocket
node per candidate. Its input is the packed external cache, not a ligand and
not a project-local path:

```bash
siteweaver-cache \
  --cache-dir /path/to/siteweaver_g1_pocket_node_cache \
  --complex-id 4ZSG \
  --out-dir /path/to/results_4ZSG \
  --mode pocket \
  --device cuda
```

The command writes a pocket score CSV and a JSON record containing the highest
scoring pocket and its nearby residue indices. The corresponding cache format
is a single `data.packed.ptzst` plus `manifest.json`; the complete cache is
distributed separately because it is about 4.7 GB. The R0 residue ranker can
be run on the matching full-atom cache with `--mode residue`.

## Optional FPocket and PRS adapters

FPocket and ProDy are intentionally not bundled as third-party executables.
The package does provide the exact invocation and feature-generation adapters
used for the study.

Run FPocket on one structure, or on a CSV containing `complex_id,pdb_path`:

```bash
siteweaver-fpocket \
  --fpocket-bin /path/to/fpocket \
  --pdb /path/to/protein.pdb \
  --out-dir /path/to/fpocket_outputs
```

For a batch, replace `--pdb` with `--manifest manifest.csv`. The adapter
copies each PDB into `<out-dir>/<complex_id>/`, runs `fpocket -f <local>.pdb`,
expects `<complex_id>_out/pockets/`, uses eight workers by default, reuses
completed outputs, and records the command, input hash, FPocket probe output,
and log path in `fpocket_run.json`.

Generate the three PRS channels after producing an active-probability table:

```bash
siteweaver-prs \
  --pdb /path/to/protein.pdb \
  --active-scores /path/to/active_context_scores.csv \
  --out-dir /path/to/prs_output
```

The active-score CSV must contain `chain_id,residue_number,insertion_code`
and `probability`. The default parameters reproduce the project workflow:
protein `CA` atoms only, ANM cutoff 15 A, gamma 1.0, 20 modes, dense ProDy
calculation up to 1,200 residues, and the sparse low-rank fallback above that
size. Outputs are `prs_features.csv`, `prs_features.npy`, and
`prs_metadata.json`; the three columns are active-to-residue response,
residue-to-active response, and their directional difference. To reproduce
the allosteric path exactly, create the CSV with:

```bash
siteweaver --pdb /path/to/protein.pdb \
  --out-dir /path/to/active_context_output \
  --task active_context
```

Changing `--cutoff`, `--gamma`, `--n-modes`, or `--chain` is supported and is
recorded in the metadata. The adapters never assume a dataset path and never
silently substitute a missing external executable.

Cache records already contain graph features and the precomputed upstream
channels. `r0_prs_with_pocket_v1` contains the three directional CA-only PRS
channels and is the recommended starting point for PRS parameter studies.
The split registry is an independent two-column CSV (`complex_id,split`), so
one cache can support multiple train/validation/test assignments.

## Bundled profiles

The exact model settings and checkpoint provenance are in
`configs/final_profiles.yaml` and `weights/manifest.json`.

- `ligand_free_pocket_final.ckpt`: 8-layer ligand-free pocket predictor.
- `active_site_final.ckpt`: active-site predictor with pocket probability.
- `cryptic_site_final.ckpt`: cryptic-site predictor with pocket probability.
- `allosteric_residue_rank_final.ckpt`: `SW-Allo-R`, the 16-layer full-atom
  residue ranker used for the direct-residue comparison.
- `allosteric_pocket_rank_final.ckpt`: `SW-Allo-P/G1`, the 16-layer FPocket
  pocket-node ranker used for the production pocket-ranking workflow.
- `active_context_final.ckpt` and `allosteric_prs_research_epoch98.ckpt`:
  optional upstream/research profiles for PRS experiments.

Every checkpoint is frozen at the selected epoch and has a recorded SHA-256.
Historical checkpoints, W&B runs, training logs, raw datasets, and caches are
intentionally excluded from this repository. Their logical names, sizes, and
hashes are recorded in `release/artifact_manifest.json` for a separate Zenodo
or institutional data upload.

## Replacing a checkpoint

Pass `--weights-dir /path/to/weights` only when replacing a bundled runtime
profile with a checkpoint using the same architecture and filename. Validate
the file with the SHA-256 manifest before sharing a result. The package does
not auto-discover checkpoints outside its own `weights/` directory.

## Provenance

The graph backbone follows the YuelPocket implementation, with SiteWeaver
task heads and the allosteric pocket-node ranker defined in this package.
Preserve the upstream attribution and applicable license terms when
redistributing the package.
