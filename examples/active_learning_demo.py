"""Compare compound-selection policies over simulated design-make-test cycles.

    python examples/active_learning_demo.py

Swap `oracle` for a function that reads your LIMS and this is the real thing.
"""
import logging

import numpy as np

from dd_pipeline.chem.standardize import standardize_table
from dd_pipeline.config import PipelineConfig
from dd_pipeline.data.simulate import ASSAY_NOISE, build_virtual_library, truth_for_smiles
from dd_pipeline.registry import Run, setup_logging
from dd_pipeline.stages.active_learning import compare_policies


def main() -> None:
    setup_logging(level=logging.INFO)
    for noisy in ("dd_pipeline.chem.splits", "dd_pipeline.models.qsar",
                  "dd_pipeline.models.applicability", "dd_pipeline.scoring.pareto",
                  "dd_pipeline.chem.featurize"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = PipelineConfig(project="active-learning-demo")
    cfg.activity_model.n_estimators = 300
    cfg.activity_model.n_ensemble = 4
    cfg.activity_model.early_stopping_rounds = None

    run = Run.create(cfg.run_root, cfg.project)
    library = build_virtual_library(n=2000, seed=cfg.seed)
    pool, _ = standardize_table(library, smiles_col="smiles")

    truth = (truth_for_smiles(pool["canonical_smiles"].tolist())
             .set_index("smiles")["true_pIC50"].to_dict())
    rng = np.random.default_rng(cfg.seed)

    def oracle(smiles: list[str]) -> np.ndarray:
        """Stands in for the assay. Note the noise: a perfect oracle would
        make the loop look far better than any real programme ever does."""
        return np.array(
            [truth.get(s, 5.0) + rng.normal(0, ASSAY_NOISE["pIC50"]) for s in smiles]
        )

    history = compare_policies(
        pool, oracle, cfg, run=run,
        policies=("random", "greedy", "ucb", "diverse_ucb"),
        n_rounds=5, batch_size=48, n_initial=96,
    )

    print("\nMean measured potency per round:\n")
    print(history.pivot_table(index="round", columns="policy",
                              values="batch_mean_measured").round(3).to_string())
    print("\nUnique scaffolds per batch:\n")
    print(history.pivot_table(index="round", columns="policy",
                              values="batch_unique_scaffolds").round(1).to_string())
    run.finalize(cfg.model_dump(mode="json"))
    print(f"\nrun directory: {run.root}")


if __name__ == "__main__":
    main()
