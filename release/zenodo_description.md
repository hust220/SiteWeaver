# SiteWeaver 0.2.0

SiteWeaver provides frozen models and portable inference for ligand-free pocket, active-site, allosteric-site, and cryptic-site prediction. The article-main allosteric modes are SW-Allo-R, a heavy-atom residue ranker, and SW-Allo-P, an FPocket candidate ranker. Both require annotated active-site residues at inference time. FPocket is installed separately. PRS is optional for historical controls and is not an input to these allosteric branches.

This software release includes five frozen checkpoints: pocket, active site, cryptic site, SW-Allo-R, and SW-Allo-P. Training caches are distributed separately and identified in `release/artifact_manifest.json`. The historical I01 cascade is preserved by the pre-switch commit but is not a current article-main model. See `README.md` for installation, input formats, training, and examples.
