# Collaborator Handoff

- `weights/allosteric_residue_rank_final.ckpt` is the no-PRS `SW-Allo-R` R0 residue ranker.
- `weights/allosteric_pocket_rank_final.ckpt` is the no-PRS `SW-Allo-P` G1 FPocket pocket-node ranker.
- `weights/allosteric_i01_prs_final.ckpt` is the article-main `I01_NOYP_PRS`
  predicted-active cascade; `predict.py --task allosteric` uses it by default.
- `training/train_sw_allo_r.py` and `training/train_sw_allo_p.py` retrain the two models from externally supplied caches.
- `data/active_site_probabilities_1160_v1.csv.gz` contains the residue-aligned active-site probabilities used by the shared data preparation.
- The default allosteric inference computes the three CA-only PRS channels
  internally. Collaborator retraining with changed PRS features still requires
  a separately named cache and experiment.

The large caches are distributed separately because they are several gigabytes. The training scripts require an explicit cache directory and two-column `complex_id,split` registry; they do not use project-local or machine-specific paths. Both scripts save `newest_completed.ckpt`, `last.ckpt`, `best.ckpt`, and `run_summary.json`, and support optional W&B logging and checkpoint resume.
