# SiteWeaver: self-contained protein site prediction and pocket ranking

SiteWeaver is a self-contained research software package for protein binding-pocket, active-site, allosteric-site, and cryptic-site prediction. It contains the model source code, selected frozen checkpoints, configuration metadata, an example structure, and reproducible adapters for the optional FPocket and ProDy PRS tools. The package does not depend on the authors' project repository and does not include machine-specific paths.

The residue-level paths run from a protein PDB and write per-residue scores to CSV files and PDB B-factors for visualization. The default article-main allosteric path is the `I01_NOYP_PRS` predicted-active cascade, which computes three CA-only PRS channels at inference time. The Figure 2 pocket-first route ranks FPocket candidate pockets with a 16-layer full-protein graph neural network and one pocket node per candidate. Its large preprocessed cache is distributed separately and is identified, together with its SHA-256 checksum, in `release/artifact_manifest.json`.

The archive includes seven frozen checkpoint files: the ligand-free pocket predictor, active-site predictor, cryptic-site predictor, the article-main I01 allosteric model, the direct residue-ranking R0 allosteric model, the FPocket pocket-node allosteric model, and the upstream active-context model. Exact filenames, epochs, architectures, feature dimensions, and SHA-256 checksums are recorded in `weights/manifest.json`.

FPocket and ProDy are not redistributed in this archive. `run_fpocket.py` reproduces the study's local-copy and `fpocket -f` invocation policy, while `compute_prs.py` reproduces the CA-only ProDy ANM/PRS feature generation with the documented cutoff, mode count, dense/sparse policy, and three directional output channels.

## Contents

- `src/siteweaver/`: model, graph, PDB, PRS, cache, and inference code.
- `predict.py`: direct PDB inference launcher.
- `run_fpocket.py`: optional FPocket launcher.
- `compute_prs.py`: optional ProDy PRS feature generator.
- `rank_cached.py`: allosteric ranking from an external packed cache.
- `weights/`: selected frozen checkpoints and their checksum manifest.
- `configs/`: final model profiles and PRS parameter template.
- `README.md`: installation, usage, input/output, and reproducibility instructions.

This release is intended for research use. Install PyTorch, NumPy, SciPy,
ProDy, and Zstandard according to `requirements-runtime.txt`; use a CUDA build
of PyTorch when GPU inference is desired.
