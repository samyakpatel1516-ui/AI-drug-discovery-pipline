"""Tests for models, scoring and the target-identification statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dd_pipeline.chem.featurize import Featurizer
from dd_pipeline.config import LeadOptConfig, ModelConfig, ObjectiveSpec, TargetIDConfig
from dd_pipeline.models.applicability import ApplicabilityDomain
from dd_pipeline.models.evaluate import bedroc, enrichment_factor
from dd_pipeline.models.qsar import EnsembleQSAR
from dd_pipeline.scoring.desirability import (
    MPOScorer,
    desirability,
    ligand_efficiency,
)
from dd_pipeline.scoring.pareto import diverse_pick, pareto_front, pareto_ranks
from dd_pipeline.targets.druggability import assess_druggability, recommend_modality
from dd_pipeline.targets.omics import benjamini_hochberg, differential_expression
from dd_pipeline.targets.rank import score_targets


# ---- QSAR ----------------------------------------------------------------
@pytest.fixture
def toy_regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 20)).astype(np.float32)
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2] + rng.normal(0, 0.2, 300)
    return X, y


def test_ensemble_beats_the_constant_baseline(toy_regression):
    """Judged against the constant predictor, not an absolute RMSE.

    Gradient-boosted trees fit a linear function only piecewise, so an
    absolute threshold here would be a test of tree geometry rather than of
    this code. Beating the mean by a clear margin is the claim that matters.
    """
    X, y = toy_regression
    m = EnsembleQSAR(ModelConfig(n_ensemble=3, n_estimators=120, early_stopping_rounds=None))
    m.fit(X[:200], y[:200], X[200:260], y[200:260])
    pred = m.predict(X[260:])
    rmse = float(np.sqrt(np.mean((y[260:] - pred.mean) ** 2)))
    baseline = float(np.sqrt(np.mean((y[260:] - y[:200].mean()) ** 2)))
    assert rmse < 0.7 * baseline, f"rmse={rmse:.3f} vs constant baseline {baseline:.3f}"


def test_ensemble_produces_nonzero_spread_and_intervals(toy_regression):
    X, y = toy_regression
    m = EnsembleQSAR(ModelConfig(n_ensemble=5, n_estimators=120, early_stopping_rounds=None))
    m.fit(X[:200], y[:200], X[200:260], y[200:260])
    pred = m.predict(X[260:])
    assert pred.std.min() >= 0
    assert pred.std.mean() > 0, "ensemble members are identical; uncertainty is meaningless"
    assert pred.lower is not None and np.all(pred.lower <= pred.upper)


def test_conformal_coverage_is_roughly_nominal(toy_regression):
    X, y = toy_regression
    m = EnsembleQSAR(ModelConfig(n_ensemble=5, n_estimators=200, early_stopping_rounds=None))
    m.fit(X[:180], y[:180], X[180:250], y[180:250])
    pred = m.predict(X[250:], level=0.9)
    cov = float(np.mean((y[250:] >= pred.lower) & (y[250:] <= pred.upper)))
    assert cov > 0.70, f"90% intervals covered only {cov:.0%} of truths"


def test_model_round_trips_through_disk(tmp_path, toy_regression):
    X, y = toy_regression
    m = EnsembleQSAR(ModelConfig(n_ensemble=2, n_estimators=60, early_stopping_rounds=None),
                     endpoint="pIC50")
    m.fit(X[:200], y[:200], X[200:260], y[200:260])
    before = m.predict(X[260:]).mean
    path = m.save(tmp_path / "m.pkl")
    after = EnsembleQSAR.load(path).predict(X[260:]).mean
    assert np.allclose(before, after)
    assert (tmp_path / "m.card.json").exists()


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        EnsembleQSAR().predict(np.zeros((2, 5), dtype=np.float32))


# ---- applicability domain ------------------------------------------------
def test_near_training_chemistry_is_in_domain_and_distant_is_not():
    train = ["c1ccc(cc1)C(=O)NC", "c1ccc(cc1)C(=O)NCC", "c1ccc(cc1)C(=O)NCCC",
             "c1ccc(cc1)C(=O)NCCCC", "c1ccc(cc1)C(=O)NC(C)C"]
    ad = ApplicabilityDomain(featurizer=Featurizer()).fit(train)
    res = ad.assess(["c1ccc(cc1)C(=O)NCCCCC", "C1CC2CCC1C2"])
    assert res["ad_mean_similarity"][0] > res["ad_mean_similarity"][1]
    assert res["in_domain"][0]
    assert not res["in_domain"][1]


# ---- ranking metrics -----------------------------------------------------
def test_enrichment_factor_perfect_and_random():
    y = np.array([1] * 10 + [0] * 90)
    perfect = np.concatenate([np.ones(10) * 10, np.arange(90) * 0.001])
    assert enrichment_factor(y, perfect, 0.10) == pytest.approx(10.0)
    # A constant score cannot enrich; argsort ties give the first k entries.
    ef_rand = enrichment_factor(y, np.linspace(1, 0, 100)[::-1], 0.10)
    assert ef_rand < 2.0


def test_bedroc_is_bounded_and_rewards_early_recognition():
    y = np.array([1] * 10 + [0] * 190)
    early = np.concatenate([np.ones(10) * 5, np.zeros(190)])
    late = np.concatenate([np.zeros(190), np.ones(10) * 5])
    b_early, b_late = bedroc(y, early), bedroc(y, late)
    assert 0.0 <= b_late < b_early <= 1.0001


# ---- desirability / MPO --------------------------------------------------
def test_desirability_is_monotone_and_bounded():
    spec = ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=6.0, high=9.0)
    d = desirability(np.array([4.0, 6.0, 7.5, 9.0, 11.0]), spec)
    assert np.all(np.diff(d) >= 0)
    assert d.min() >= 0 and d.max() <= 1
    assert d[0] < 0.05 and d[-1] > 0.95


def test_lower_better_is_the_mirror_image():
    spec = ObjectiveSpec(endpoint="clint", kind="lower_better", low=10.0, high=60.0)
    d = desirability(np.array([5.0, 35.0, 100.0]), spec)
    assert d[0] > d[1] > d[2]


def test_window_desirability_peaks_inside_the_window():
    spec = ObjectiveSpec(endpoint="logD", kind="window", low=1.0, high=3.5)
    d = desirability(np.array([-1.0, 2.2, 6.0]), spec)
    assert d[1] > d[0] and d[1] > d[2]
    assert d[1] > 0.9


def test_missing_values_score_zero_not_a_guess():
    spec = ObjectiveSpec(endpoint="logS", kind="higher_better", low=-6.0, high=-3.5)
    assert desirability(np.array([np.nan, np.inf, -4.0]), spec)[0] == 0.0
    assert desirability(np.array([np.nan]), spec)[0] == 0.0


def test_critical_failure_zeroes_the_whole_mpo_score():
    cfg = LeadOptConfig(objectives=[
        ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=6.0, high=9.0,
                      weight=1.0, critical=True),
        ObjectiveSpec(endpoint="logS", kind="higher_better", low=-6.0, high=-3.5, weight=1.0),
    ])
    df = pd.DataFrame({"pIC50": [9.0, 3.0], "logS": [-3.5, -3.5]})
    res = MPOScorer(cfg).score_frame(df)
    assert res.score[0] > 0.9
    assert res.score[1] == 0.0
    assert res.failed_critical[1]


def test_geometric_mean_punishes_a_single_bad_term():
    cfg = LeadOptConfig(objectives=[
        ObjectiveSpec(endpoint="a", kind="higher_better", low=0.0, high=1.0),
        ObjectiveSpec(endpoint="b", kind="higher_better", low=0.0, high=1.0),
    ])
    df = pd.DataFrame({"a": [1.0, 0.5], "b": [0.02, 0.5]})
    s = MPOScorer(cfg).score_frame(df).score
    assert s[1] > s[0], "a balanced compound must beat one that fails an objective"


def test_uncertainty_penalty_separates_confident_from_uncertain():
    """Two molecules with identical predicted means must not score the same
    when one of them is a guess. Without this, a generative optimiser is
    rewarded for walking into the region where the model is most wrong."""
    from dd_pipeline.scoring.desirability import pessimistic_frame

    objs = [
        ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=6.0, high=9.0),
        ObjectiveSpec(endpoint="hlm_clint", kind="lower_better", low=10.0, high=60.0),
        ObjectiveSpec(endpoint="logD", kind="window", low=1.0, high=3.5),
    ]
    df = pd.DataFrame({
        "pIC50": [8.0, 8.0], "pIC50_std": [0.1, 1.2],
        "hlm_clint": [20.0, 20.0], "hlm_clint_std": [2.0, 25.0],
        "logD": [2.0, 2.0], "logD_std": [0.1, 1.5],
    })
    scorer = MPOScorer(LeadOptConfig(objectives=objs))
    optimistic = scorer.score_frame(df).score
    assert optimistic[0] == pytest.approx(optimistic[1]), "means are identical by construction"

    conservative = scorer.score_frame(pessimistic_frame(df, objs, beta=1.0)).score
    assert conservative[0] > conservative[1], "the uncertain molecule must score lower"
    assert conservative[1] < 0.5 * conservative[0]


def test_uncertainty_penalty_shifts_each_objective_the_right_way():
    from dd_pipeline.scoring.desirability import pessimistic_frame

    objs = [
        ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=6.0, high=9.0),
        ObjectiveSpec(endpoint="clint", kind="lower_better", low=10.0, high=60.0),
        ObjectiveSpec(endpoint="logD", kind="window", low=1.0, high=3.0),
    ]
    df = pd.DataFrame({
        "pIC50": [8.0], "pIC50_std": [0.5],
        "clint": [20.0], "clint_std": [5.0],
        "logD": [2.5], "logD_std": [0.4],   # above the window centre (2.0)
    })
    out = pessimistic_frame(df, objs, beta=1.0)
    assert out.loc[0, "pIC50"] == pytest.approx(7.5)    # potency assumed worse
    assert out.loc[0, "clint"] == pytest.approx(25.0)   # clearance assumed worse
    assert out.loc[0, "logD"] == pytest.approx(2.9)     # pushed away from centre


def test_zero_beta_recovers_optimistic_scoring():
    from dd_pipeline.scoring.desirability import pessimistic_frame

    objs = [ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=6.0, high=9.0)]
    df = pd.DataFrame({"pIC50": [8.0], "pIC50_std": [1.0]})
    assert pessimistic_frame(df, objs, beta=0.0).equals(df)


def test_uncertainty_penalty_tolerates_missing_std_columns():
    from dd_pipeline.scoring.desirability import pessimistic_frame

    objs = [ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=6.0, high=9.0)]
    df = pd.DataFrame({"pIC50": [8.0]})
    assert pessimistic_frame(df, objs, beta=1.0).loc[0, "pIC50"] == 8.0


def test_ligand_efficiency_penalises_molecular_weight():
    le = ligand_efficiency(np.array([8.0, 8.0]), np.array([20.0, 50.0]))
    assert le[0] > le[1]


# ---- Pareto --------------------------------------------------------------
def test_pareto_front_identifies_non_dominated_points():
    # (potency, solubility), both maximised. Point 3 is dominated by point 0.
    obj = np.array([[9.0, 9.0], [9.0, 1.0], [1.0, 9.0], [2.0, 2.0]])
    front = pareto_front(obj, [True, True])
    assert front.tolist() == [True, False, False, False]


def test_pareto_respects_minimisation():
    obj = np.array([[9.0, 1.0], [9.0, 9.0]])   # second column minimised
    front = pareto_front(obj, [True, False])
    assert front.tolist() == [True, False]


def test_pareto_over_desirabilities_handles_window_objectives():
    """Regression test. A `window` objective has no single direction: logD 2.0
    beats both 0.5 and 6.0. Running the front over raw endpoint values with
    maximize=True declared the over-lipophilic compound non-dominated and threw
    away the good one. The front must be computed over desirabilities."""
    cfg = LeadOptConfig(objectives=[
        ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=6.0, high=9.0),
        ObjectiveSpec(endpoint="logD", kind="window", low=1.0, high=3.5),
    ])
    df = pd.DataFrame({"pIC50": [8.0, 8.0], "logD": [2.0, 6.0]})
    ann = MPOScorer(cfg).annotate(df)

    good = pareto_front(ann[["d_pIC50", "d_logD"]].to_numpy(float), [True, True])
    assert good.tolist() == [True, False], "the in-window compound must dominate"

    naive = pareto_front(df[["pIC50", "logD"]].to_numpy(float), [True, True])
    assert naive.tolist() == [False, True], "raw-value front is wrong, as expected"


def test_pareto_ranks_are_layered():
    obj = np.array([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]])
    assert pareto_ranks(obj, [True, True]).tolist() == [1, 2, 3]


def test_diverse_pick_enforces_the_similarity_cap():
    f = Featurizer()
    # Three near-identical amides plus two unrelated scaffolds.
    smis = ["c1ccc(cc1)C(=O)NC", "c1ccc(cc1)C(=O)NCC", "c1ccc(cc1)C(=O)NCCC",
            "C1CCNCC1", "c1ccc2[nH]ccc2c1"]
    fps = f.bit_fingerprints(smis)
    picked = diverse_pick(fps, np.array([5.0, 4.9, 4.8, 1.0, 0.9]), n=5, max_similarity=0.6)
    assert len(picked) < 5, "the similarity cap was not applied"
    assert 0 in picked.tolist()


# ---- target identification statistics -----------------------------------
def test_benjamini_hochberg_is_monotone_and_conservative():
    p = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
    q = benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    assert q.max() <= 1.0


def test_differential_expression_recovers_a_planted_gene():
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(50)]
    cols = [f"S{i}" for i in range(20)]
    data = rng.lognormal(2.0, 0.3, size=(50, 20))
    data[7, :10] *= 8.0                                   # strongly up in cases
    expr = pd.DataFrame(data, index=genes, columns=cols)
    groups = pd.Series(["case"] * 10 + ["control"] * 10, index=cols)
    res = differential_expression(expr, groups, "case", "control")
    top = res.iloc[0]
    assert top["gene"] == "G7"
    assert top["log2fc"] > 1.5
    assert top["fdr"] < 0.01


def test_differential_expression_requires_replicates():
    expr = pd.DataFrame(np.ones((5, 2)), index=list("ABCDE"), columns=["a", "b"])
    groups = pd.Series(["case", "control"], index=["a", "b"])
    with pytest.raises(ValueError):
        differential_expression(expr, groups, "case", "control")


def test_pan_essential_genes_are_penalised_in_target_scoring():
    tbl = pd.DataFrame({
        "gene": ["SELECTIVE", "PAN_ESSENTIAL"],
        "log2fc": [3.0, 3.0],
        "fdr": [1e-6, 1e-6],
        "dep_mean_disease": [-1.2, -1.2],
        "dep_selectivity": [1.0, 0.0],
        "is_common_essential": [False, True],
        "druggability": [0.8, 0.8],
        "safety": [0.9, 0.3],
        "ligand_evidence": [0.5, 0.5],
        "family": ["kinase", "kinase"],
    })
    scored = score_targets(tbl, TargetIDConfig())
    assert scored.iloc[0]["gene"] == "SELECTIVE"
    assert "pan-essential" in scored[scored.gene == "PAN_ESSENTIAL"].iloc[0]["rationale"]


def test_target_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        TargetIDConfig(weights={"dependency": 0.5, "druggability": 0.2})


def test_druggability_falls_back_to_family_prior_without_structure():
    ann = pd.DataFrame({"gene": ["A"], "family": ["kinase"]})
    out = assess_druggability(ann)
    assert out.loc[0, "druggability_source"] == "family_prior"
    assert out.loc[0, "druggability"] == pytest.approx(0.95)


def test_modality_recommendation_avoids_small_molecules_for_pocketless_tf():
    row = pd.Series({"druggability": 0.1, "family": "transcription_factor",
                     "has_degrader_handle": True})
    assert recommend_modality(row) == "targeted_degrader"
    assert recommend_modality(pd.Series({"druggability": 0.8, "family": "kinase"})) \
        == "small_molecule"
