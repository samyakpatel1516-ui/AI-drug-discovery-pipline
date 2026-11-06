"""Tests for the public-benchmark loaders and the domain-coverage guard.

Network-dependent tests are marked `netlocal`: they pass when the datasets are
already cached (the normal case after one run) and skip otherwise, so the suite
stays usable offline.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import pytest

from dd_pipeline.data.public import DATASETS, build_decoy_library, load_dataset

CACHE = Path("data/public")

# Download failures should skip, not fail: these tests depend on a third-party
# host being up, which is not what they are testing. Set DD_REQUIRE_NETWORK=1
# to turn a download failure into a real failure instead -- worth doing in CI,
# where silently skipping the real-data path would hide a broken loader.
_REQUIRE_NETWORK = os.environ.get("DD_REQUIRE_NETWORK") == "1"
_fetch_failures: dict[str, str] = {}


def _try_load(name: str):
    """Load a dataset, caching across tests; skip the test if it is unreachable."""
    if name in _fetch_failures and not _REQUIRE_NETWORK:
        pytest.skip(f"{name} unreachable: {_fetch_failures[name]}")
    try:
        return load_dataset(name, cache_dir=CACHE)
    except (URLError, HTTPError, TimeoutError, OSError) as exc:
        _fetch_failures[name] = str(exc)
        if _REQUIRE_NETWORK:
            raise
        pytest.skip(f"{name} could not be downloaded ({exc}); set DD_REQUIRE_NETWORK=1 "
                    "to make this a failure")


def test_dataset_specs_are_self_consistent():
    for name, spec in DATASETS.items():
        assert spec.name == name
        assert spec.url.startswith("https://")
        assert spec.citation and spec.description
        assert spec.endpoint


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_dataset("not_a_dataset")


@pytest.mark.parametrize("name,expected_endpoint,min_rows", [
    ("bace", "pIC50", 1400),
    ("lipophilicity", "logD", 4000),
    ("esol", "logS", 950),
])
def test_loader_returns_curated_frame(name, expected_endpoint, min_rows):
    df, prov = _try_load(name)
    assert expected_endpoint in df.columns
    assert "canonical_smiles" in df.columns
    assert len(df) >= min_rows
    assert df[expected_endpoint].notna().all()
    # Standardisation must leave no duplicate structures behind -- this is the
    # train/test leakage guard, and these published files really do contain
    # duplicates.
    assert df["canonical_smiles"].nunique() == len(df)
    assert prov["file_sha256"] and len(prov["file_sha256"]) == 64
    assert prov["n_rows_final"] == len(df)


def test_bace_potency_values_are_physically_plausible():
    df, _ = _try_load("bace")
    p = df["pIC50"]
    # pIC50 2-11 spans 10 mM to 10 pM; anything outside is a unit error.
    assert p.min() > 1.0 and p.max() < 12.0
    assert 5.5 < p.mean() < 7.5


def test_esol_excludes_sub_fragment_molecules():
    """The 106 compounds dropped from ESOL are below 6 heavy atoms.

    Documented as a test because it is a property of the *benchmark*, not a
    bug: ~9% of a standard solubility set is industrial solvents, which bounds
    how much a model trained on it can say about drug-like chemistry.
    """
    from rdkit import Chem

    df, prov = _try_load("esol")
    assert prov["n_rows_raw"] - prov["n_rows_final"] > 50
    sizes = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                      for s in df["canonical_smiles"]])
    assert sizes.min() >= 6


def test_decoy_library_mixes_roles_and_excludes_actives():
    actives = pd.DataFrame({"canonical_smiles": [
        "CC(=O)Nc1ccc(O)cc1", "Cc1ccc(cc1)S(=O)(=O)Nc1ccccn1",
        "CN1CCN(CC1)c1ncc2cc(F)ccc2n1", "O=C(Nc1ccccc1)N1CCN(c2ccccn2)CC1",
    ]})
    pool = pd.DataFrame({"canonical_smiles": [
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
        "CN(C)CCOC(c1ccccc1)c1ccccc1", "Cc1ccc2nc(N)sc2c1",
        "O=C1N(c2ccccc2)C(=O)c2ccccc21", "NC(=O)c1ccc(cc1)N1CCN(C)CC1",
        "CC(=O)Nc1ccc(O)cc1",   # duplicate of an active -- must be excluded
    ]})
    lib = build_decoy_library(actives, pool, n_decoys_per_active=1, seed=0)
    roles = set(lib["library_role"])
    assert roles == {"measured", "presumed_inactive_decoy"}
    decoys = lib[lib["library_role"] == "presumed_inactive_decoy"]["canonical_smiles"]
    assert not set(decoys) & set(actives["canonical_smiles"])
    assert lib["compound_id"].nunique() == len(lib)


def test_decoy_library_is_reproducible():
    actives = pd.DataFrame({"canonical_smiles": ["CC(=O)Nc1ccc(O)cc1", "Cc1ccc2nc(N)sc2c1"]})
    pool = pd.DataFrame({"canonical_smiles": [
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
        "CN(C)CCOC(c1ccccc1)c1ccccc1", "O=C1N(c2ccccc2)C(=O)c2ccccc21",
        "NC(=O)c1ccc(cc1)N1CCN(C)CC1",
    ]})
    a = build_decoy_library(actives, pool, n_decoys_per_active=2, seed=7)
    b = build_decoy_library(actives, pool, n_decoys_per_active=2, seed=7)
    assert a["canonical_smiles"].tolist() == b["canonical_smiles"].tolist()


def test_out_of_domain_objective_is_reported_not_silently_scored(caplog):
    """Regression test for the real bug found on BACE-1 + ESOL.

    An out-of-domain prediction is a confident-looking extrapolation, not a
    missing value, so desirability scores it like any other number. The
    pipeline must surface per-objective domain coverage instead of returning a
    healthy-looking combined score built on extrapolation.
    """
    import logging

    from dd_pipeline.config import ObjectiveSpec
    from dd_pipeline.stages.lead_opt import check_objective_domain_coverage

    objectives = [
        ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=7.0, high=9.5),
        ObjectiveSpec(endpoint="logD", kind="window", low=1.0, high=3.5),
        ObjectiveSpec(endpoint="logS", kind="higher_better", low=-5.5, high=-3.5),
    ]
    frame = pd.DataFrame({
        "pIC50_in_domain": [True] * 10,
        "logD_in_domain": [True] * 4 + [False] * 6,     # 40% -> caution
        "logS_in_domain": [False] * 10,                  # 0%  -> critical
    })

    with caplog.at_level(logging.WARNING):
        cov = check_objective_domain_coverage(frame, objectives, label="test set")

    assert cov == pytest.approx({"pIC50": 1.0, "logD": 0.4, "logS": 0.0})
    levels = {r.levelno for r in caplog.records}
    assert logging.ERROR in levels, "0% domain coverage must be an error, not a warning"
    assert logging.WARNING in levels
    text = caplog.text
    assert "logS" in text and "logD" in text
    assert "pIC50" not in text, "a fully in-domain objective should not be flagged"


def test_domain_coverage_ignores_objectives_without_a_domain_model():
    from dd_pipeline.config import ObjectiveSpec
    from dd_pipeline.stages.lead_opt import check_objective_domain_coverage

    objectives = [ObjectiveSpec(endpoint="hlm_clint", kind="lower_better",
                                low=10.0, high=60.0)]
    cov = check_objective_domain_coverage(pd.DataFrame({"x": [1, 2]}), objectives)
    assert cov == {}
