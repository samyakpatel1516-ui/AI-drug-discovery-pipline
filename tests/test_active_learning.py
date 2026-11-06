"""Tests for the design-make-test-analyse loop."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dd_pipeline.chem.featurize import Featurizer
from dd_pipeline.config import PipelineConfig
from dd_pipeline.stages.active_learning import (
    POLICIES,
    greedy_policy,
    run_active_learning,
    ucb_policy,
)


@pytest.fixture
def small_cfg():
    cfg = PipelineConfig()
    cfg.activity_model.n_estimators = 60
    cfg.activity_model.n_ensemble = 3
    cfg.activity_model.early_stopping_rounds = None
    return cfg


@pytest.fixture
def pool_and_oracle():
    from dd_pipeline.data.simulate import build_virtual_library, truth_for_smiles

    lib = build_virtual_library(n=320, seed=42)
    pool = lib.rename(columns={"smiles": "canonical_smiles"})
    truth = (truth_for_smiles(pool["canonical_smiles"].tolist())
             .set_index("smiles")["true_pIC50"].to_dict())

    def oracle(smis: list[str]) -> np.ndarray:
        return np.array([truth.get(s, 5.0) for s in smis], dtype=float)

    return pool, oracle


def test_greedy_policy_takes_the_top_predictions():
    df = pd.DataFrame({"pred": [1.0, 9.0, 5.0], "pred_std": [0.1, 0.1, 0.1]})
    assert greedy_policy(df, [None] * 3, 2).tolist() == [1, 2]


def test_ucb_prefers_uncertainty_when_means_are_tied():
    df = pd.DataFrame({"pred": [5.0, 5.0], "pred_std": [0.1, 2.0]})
    assert ucb_policy(df, [None] * 2, 1).tolist() == [1]


def test_ucb_with_zero_beta_is_greedy():
    df = pd.DataFrame({"pred": [1.0, 9.0], "pred_std": [5.0, 0.0]})
    assert ucb_policy(df, [None] * 2, 1, beta=0.0).tolist() == [1]


def test_all_registered_policies_return_valid_index_sets(pool_and_oracle):
    pool, _ = pool_and_oracle
    f = Featurizer()
    df = pool.head(40).copy()
    rng = np.random.default_rng(0)
    df["pred"] = rng.normal(7, 1, len(df))
    df["pred_std"] = rng.uniform(0.05, 0.5, len(df))
    df["in_domain"] = True
    fps = f.bit_fingerprints(df["canonical_smiles"].tolist())
    for name, fn in POLICIES.items():
        idx = np.asarray(fn(df, fps, 10), dtype=int)
        assert idx.size > 0, f"{name} selected nothing"
        assert idx.size <= 10, f"{name} exceeded the batch budget"
        assert len(set(idx.tolist())) == idx.size, f"{name} returned duplicates"
        assert idx.min() >= 0 and idx.max() < len(df), f"{name} returned out-of-range indices"


def test_loop_accumulates_training_data_and_never_reselects(small_cfg, pool_and_oracle):
    pool, oracle = pool_and_oracle
    res = run_active_learning(
        pool, oracle, small_cfg, n_rounds=3, batch_size=20, n_initial=60,
        policy="greedy",
    )
    hist = res["history"]
    assert len(hist) == 3
    # Training set grows by the batch size each round.
    assert hist["n_train_before"].tolist() == [60, 80, 100]
    assert len(res["train"]) == 120
    # A compound must never be measured twice.
    assert res["train"]["canonical_smiles"].nunique() == len(res["train"])


def test_model_driven_selection_beats_random(small_cfg, pool_and_oracle):
    """The point of the model. If this fails, the loop is not using it."""
    pool, oracle = pool_and_oracle
    greedy = run_active_learning(pool.copy(), oracle, small_cfg, n_rounds=2,
                                 batch_size=25, n_initial=60, policy="greedy")
    rand = run_active_learning(pool.copy(), oracle, small_cfg, n_rounds=2,
                               batch_size=25, n_initial=60, policy="random")
    assert (greedy["history"]["batch_mean_measured"].mean()
            > rand["history"]["batch_mean_measured"].mean()), \
        "greedy selection did not beat random; the model is not informing selection"


def test_diversity_policy_yields_more_scaffolds_per_batch(small_cfg, pool_and_oracle):
    pool, oracle = pool_and_oracle
    g = run_active_learning(pool.copy(), oracle, small_cfg, n_rounds=2, batch_size=25,
                            n_initial=60, policy="greedy")
    d = run_active_learning(pool.copy(), oracle, small_cfg, n_rounds=2, batch_size=25,
                            n_initial=60, policy="diverse_ucb")
    # Diversity is the explicit purpose of the policy, so this must hold even
    # where the two policies score similarly on potency.
    assert (d["history"]["batch_unique_scaffolds"].mean()
            >= g["history"]["batch_unique_scaffolds"].mean())


def test_cumulative_best_is_monotone(small_cfg, pool_and_oracle):
    """The best compound found so far can never get worse."""
    pool, oracle = pool_and_oracle
    res = run_active_learning(pool, oracle, small_cfg, n_rounds=3, batch_size=20,
                              n_initial=60, policy="ucb")
    cb = res["history"]["cumulative_best"].to_numpy()
    assert np.all(np.diff(cb) >= -1e-9)


def test_loop_requires_the_smiles_column(small_cfg):
    with pytest.raises(ValueError):
        run_active_learning(pd.DataFrame({"x": [1]}), lambda s: np.zeros(len(s)),
                            small_cfg, n_rounds=1)
