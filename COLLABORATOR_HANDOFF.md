# Collaborator handoff

The article-main allosteric models are SW-Allo-R (R0; residue ranking) and SW-Allo-P (G1; FPocket candidate ranking). Both use annotated active-site residues supplied by the user. Neither uses PRS or the historical I01 predicted-active cascade. The pre-switch I01 release is preserved by commit `0d2148b8f4446889f040b8aebf42304eb40b6afb`; I01 does not require a new ENM control for the current manuscript.

The corresponding weights are `weights/allosteric_residue_rank_final.ckpt` and `weights/allosteric_pocket_rank_final.ckpt`. `training/train_sw_allo_r.py` and `training/train_sw_allo_p.py` accept the separately distributed caches and a two-column `complex_id,split` registry. They save newest, last, and best checkpoints in run-specific directories.

FPocket is an external dependency for G1. PRS generation scripts are retained as optional collaborator tools, not as active R0/G1 inputs. Any ENM variant should use its own cache and run name; it is not interchangeable with R0/G1 without matched inputs, labels, and evaluation.

`training/train_site_labels.py` reproduces the released active-site `AS_PROB` and cryptic-site `CR_PROB` training architectures using their externally supplied caches. The frozen checkpoints are already in `weights/`. All four training caches and the shared R0/G1 split registry are separate transfer artifacts, not Git files. Cached `pdb_path` fields may refer to the original machine; use the accompanying structure bundle and resolve structures by complex ID when constructing new ENM features. The packed graphs themselves train without reading those paths.
