"""Integration tests: the stages must actually fit together and be reproducible."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from dd_pipeline.chem.enumerate import BioisostereEnumerator, GAConfig, GeneticOptimizer
from dd_pipeline.config import PipelineConfig
from dd_pipeline.data.loaders import to_pactivity
from dd_pipeline.data.simulate import (
    build_virtual_library,
    simulate_assay_data,
    simulate_omics,
    truth_for_smiles,
)
from dd_pipeline.registry import Run, frame_digest


# ---- unit conversion -----------------------------------------------------
def test_nanomolar_to_pactivity():
    p = to_pactivity(pd.Series([10.0, 1000.0]), pd.Series(["nM", "nM"]))
    assert p.iloc[0] == pytest.approx(8.0)
    assert p.iloc[1] == pytest.approx(6.0)


def test_mixed_units_are_each_converted_correctly():
    p = to_pactivity(pd.Series([10.0, 0.01]), pd.Series(["nM", "uM"]))
    assert p.iloc[0] == pytest.approx(p.iloc[1]), "10 nM and 0.01 uM are the same thing"


def test_unconvertible_values_become_nan_not_wrong_numbers():
    p = to_pactivity(pd.Series([0.0, -5.0, 10.0]), "nM")
    assert p.isna().iloc[0] and p.isna().iloc[1]
    assert not p.isna().iloc[2]


# ---- simulated data ------------------------------------------------------
def test_virtual_library_is_valid_and_unique():
    lib = build_virtual_library(n=120, seed=3)
    from rdkit import Chem
    assert len(lib) > 50
    assert lib["smiles"].nunique() == len(lib)
    assert all(Chem.MolFromSmiles(s) is not None for s in lib["smiles"])


def test_library_is_reproducible_for_a_fixed_seed():
    a = build_virtual_library(n=100, seed=11)
    b = build_virtual_library(n=100, seed=11)
    assert a["smiles"].tolist() == b["smiles"].tolist()


def test_library_is_reproducible_across_processes():
    """Regression test for a bug this suite originally missed.

    `BRICSDecompose` returns a set of SMILES strings, and iteration order over
    a set of strings depends on per-process hash randomisation. The library was
    therefore different on every invocation while looking perfectly
    reproducible within a single process -- so the in-process test above passed
    and the pipeline's provenance guarantees were quietly worthless.

    Two runs under different PYTHONHASHSEED values must agree.
    """
    import hashlib
    import subprocess
    import sys

    script = (
        "import logging; logging.basicConfig(level=logging.ERROR);"
        "from dd_pipeline.data.simulate import build_virtual_library;"
        "lib = build_virtual_library(n=80, seed=11);"
        "print(''.join(lib['smiles'].tolist()))"
    )
    digests = []
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env, cwd=os.getcwd(), timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        digests.append(hashlib.sha256(out.stdout.strip().encode()).hexdigest())
    assert len(set(digests)) == 1, (
        "library generation depends on string hash order across processes"
    )


def test_simulated_endpoints_have_learnable_structure():
    """The oracle must be a function of structure, or the whole demo is
    measuring the pipeline's ability to fit noise."""
    lib = build_virtual_library(n=300, seed=5)
    truth = truth_for_smiles(lib["smiles"].tolist())
    assert truth["true_pIC50"].std() > 0.3
    # Lipophilicity should trade off against solubility, by construction.
    r = np.corrcoef(truth["true_logS"], truth["true_logD"])[0, 1]
    assert r < -0.5, f"expected a solubility/lipophilicity trade-off, got r={r:.2f}"


def test_assay_simulation_adds_noise_and_replicates():
    lib = build_virtual_library(n=200, seed=5)
    obs = simulate_assay_data(lib, n_measured=100, n_replicates=3, seed=5)
    assert obs["compound_id"].nunique() <= 100
    assert len(obs) >= obs["compound_id"].nunique()
    merged = obs.merge(truth_for_smiles(lib["smiles"].tolist()), on="smiles")
    resid = merged["pIC50"] - merged["true_pIC50"]
    assert 0.1 < resid.std() < 0.8, "assay noise is implausible"
    assert abs(resid.mean()) < 0.15, "noise should be unbiased"


def test_omics_simulation_plants_recoverable_targets():
    om = simulate_omics(n_genes=200, n_true_hits=5, seed=5)
    key = om["answer_key"]
    assert (key["category"] == "true_target").sum() == 5
    assert (key["category"] == "decoy_pan_essential").sum() > 0
    assert om["expression"].shape[0] == 200
    assert (om["expression"] >= 0).to_numpy().all()


# ---- analog generation ---------------------------------------------------
def test_bioisostere_enumeration_produces_valid_novel_molecules():
    from rdkit import Chem
    seed = "Cc1ccc(cc1)C(=O)Nc1ccccc1"
    analogs = BioisostereEnumerator().enumerate(seed, limit=30)
    assert len(analogs) > 3
    assert all(Chem.MolFromSmiles(a.smiles) is not None for a in analogs)
    assert all(a.smiles != Chem.MolToSmiles(Chem.MolFromSmiles(seed)) for a in analogs)


def test_genetic_optimizer_improves_the_objective():
    """A GA that does not beat its starting point is not optimising anything."""
    from rdkit import Chem
    from rdkit.Chem import Crippen

    # Transparent objective: maximise clogP. Easy to verify, no models needed.
    def score(smis: list[str]) -> np.ndarray:
        out = []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            out.append(Crippen.MolLogP(m) if m is not None else -99.0)
        return np.array(out)

    seeds = ["Cc1ccc(cc1)C(=O)Nc1ccccc1", "O=C(Nc1ccccc1)N1CCN(c2ccccn2)CC1"]
    ga = GeneticOptimizer(
        score,
        GAConfig(population_size=60, n_generations=2, min_similarity_to_seed=0.2,
                 max_sa_score=6.0, novelty_weight=0.0, seed=1),
    )
    out = ga.run(seeds)
    assert out, "no analogs survived"
    assert max(a.score for a in out) > max(score(seeds)), "GA did not improve the objective"
    assert len(ga.history) >= 1


def test_genetic_optimizer_rejects_unparsable_seeds():
    with pytest.raises(ValueError):
        GeneticOptimizer(lambda s: np.zeros(len(s))).run(["not_a_molecule"])


# ---- registry / provenance ----------------------------------------------
def test_run_records_artifacts_and_hashes(tmp_path):
    run = Run.create(tmp_path, "unit-test", run_id="fixed")
    df = pd.DataFrame({"a": [1, 2, 3]})
    run.save_table(df, "t", "stage")
    run.log_metrics("stage", {"x": 1.0})
    manifest_path = run.finalize({"project": "unit-test"})
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifacts"][0]["rows"] == 3
    assert len(manifest["artifacts"][0]["sha256"]) == 64
    assert manifest["metrics"]["stage"]["x"] == 1.0
    assert (run.root / "environment.json").exists()


def test_frame_digest_ignores_column_order():
    a = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    assert frame_digest(a) == frame_digest(a[["y", "x"]])


def test_frame_digest_changes_with_content():
    a = pd.DataFrame({"x": [1, 2]})
    b = pd.DataFrame({"x": [1, 3]})
    assert frame_digest(a) != frame_digest(b)


# ---- full pipeline -------------------------------------------------------
@pytest.mark.slow
def test_full_demo_runs_and_produces_candidates(tmp_path):
    cfg = PipelineConfig(run_root=tmp_path, project="itest")
    cfg.activity_model.n_estimators = 120
    cfg.admet_model.n_estimators = 120
    cfg.activity_model.n_ensemble = 3
    cfg.admet_model.n_ensemble = 3
    cfg.hit_discovery.n_hits = 30
    cfg.lead_opt.n_seeds = 5
    cfg.lead_opt.max_generations = 1
    cfg.lead_opt.n_final_candidates = 10

    from dd_pipeline.pipeline import run_demo
    res = run_demo(cfg, run_id="itest", library_size=600, n_measured=350)

    assert len(res["hits"]) > 0
    assert len(res["lead_opt"]["candidates"]) > 0
    assert (res["run"].root / "manifest.json").exists()
    # The activity model must beat the constant baseline on the held-out split.
    m = res["metrics"]["activity_model_pIC50"]
    assert m["rmse"] < m["baseline_constant_rmse"]
    # And the screen must enrich over picking at random from the pool the
    # model actually ranked. Asserting the enrichment itself (rather than
    # comparing two rates) keeps the test honest about the denominator, which
    # is the thing that was wrong the first time round.
    o = res["oracle"]
    assert o["hit_list_true_pIC50_mean"] > 0
    assert o["screened_pool_size"] > 0
    assert o["screening_enrichment_at_6"] > 1.0, (
        f"the hit list is no better than a random pool pick: "
        f"{o['hit_list_true_hit_rate_at_6']:.3f} vs "
        f"{o['screened_pool_hit_rate_at_6']:.3f}"
    )
    # The shortlist's predicted ranking should track the simulated truth; a
    # near-zero or negative value means the optimiser exploited model error.
    if "candidate_mpo_pred_vs_true_spearman" in o:
        assert o["candidate_mpo_pred_vs_true_spearman"] > 0.0
