"""Tests for the chemistry layer.

These are the tests that catch the bugs that silently ruin a model: a
standardiser that lets duplicates through, a split that leaks scaffolds, a
featurizer whose output depends on input SMILES spelling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from dd_pipeline.chem.featurize import Featurizer
from dd_pipeline.chem.filters import apply_filters, reactive_groups
from dd_pipeline.chem.splits import murcko_scaffold, scaffold_split
from dd_pipeline.chem.standardize import Standardizer, standardize_table
from dd_pipeline.chem.synth import sa_score_smiles
from dd_pipeline.config import HitDiscoveryConfig


# ---- standardisation -----------------------------------------------------
def test_salt_is_stripped():
    st = Standardizer()
    smi, _ = st.standardize_smiles("CC(=O)Oc1ccccc1C(=O)O.[Na+]")
    assert smi is not None
    assert "Na" not in smi


def test_same_molecule_different_spellings_canonicalise_identically():
    st = Standardizer()
    a, _ = st.standardize_smiles("c1ccccc1C(=O)O")
    b, _ = st.standardize_smiles("OC(=O)c1ccccc1")
    assert a == b


def test_metal_containing_structures_are_rejected():
    st = Standardizer()
    smi, _ = st.standardize_smiles("[Pt](Cl)(Cl)(N)N")
    assert smi is None


def test_unparsable_smiles_returns_none():
    st = Standardizer()
    assert st.standardize_smiles("not_a_molecule")[0] is None
    assert st.standardize_smiles("C1CC")[0] is None


def test_replicates_are_aggregated_and_spread_recorded():
    df = pd.DataFrame({
        # Same molecule twice, spelled differently, plus one distinct one.
        "smiles": ["c1ccccc1C(=O)O", "OC(=O)c1ccccc1", "CC(=O)Nc1ccc(O)cc1"],
        "pIC50": [7.0, 8.0, 5.0],
    })
    out, stats = standardize_table(df, value_col="pIC50")
    assert len(out) == 2, "the two benzoic acid spellings must collapse to one row"
    assert stats.n_duplicate_groups == 1
    row = out[out["n_replicates"] == 2].iloc[0]
    assert row["pIC50"] == pytest.approx(7.5)      # median of 7 and 8
    assert row["pIC50_spread"] == pytest.approx(1.0)
    assert row["n_replicates"] == 2


# ---- featurisation -------------------------------------------------------
def test_featurizer_is_invariant_to_smiles_spelling():
    f = Featurizer()
    fm = f.transform(["c1ccccc1C(=O)O", "OC(=O)c1ccccc1"])
    assert np.allclose(fm.X[0], fm.X[1])


def test_featurizer_shape_and_validity_mask():
    f = Featurizer()
    fm = f.transform(["CCO", "garbage", "c1ccccc1"])
    assert fm.X.shape == (3, fm.n_features)
    assert fm.valid_mask.tolist() == [True, False, True]
    assert np.allclose(fm.X[1], 0.0)


def test_descriptor_block_is_finite_for_odd_molecules():
    f = Featurizer()
    fm = f.transform(["[O-]S(=O)(=O)[O-]", "C", "N#N", "FC(F)(F)C(F)(F)F"])
    assert np.isfinite(fm.X).all()


# ---- splitting -----------------------------------------------------------
def test_scaffold_split_does_not_leak_scaffolds():
    # Two distinct scaffold families, several analogs each.
    smiles = (
        [f"c1ccc(cc1)C(=O)N{'C' * i}" for i in range(1, 9)]
        + [f"c1ccc2[nH]ccc2c1{'C' * i}" for i in range(1, 9)]
        + [f"C1CCN(CC1)C(=O)C{'C' * i}" for i in range(1, 9)]
    )
    tr, va, te = scaffold_split(smiles, test_frac=0.3, valid_frac=0.1, seed=0)
    s_tr = {murcko_scaffold(smiles[i]) for i in tr}
    s_te = {murcko_scaffold(smiles[i]) for i in te}
    assert not (s_tr & s_te), "a scaffold appears in both train and test"


def test_split_partitions_every_index_exactly_once():
    smiles = [f"c1ccc(cc1)C(=O)N{'C' * i}" for i in range(1, 30)]
    tr, va, te = scaffold_split(smiles, 0.2, 0.1, seed=0)
    allidx = np.concatenate([tr, va, te])
    assert sorted(allidx.tolist()) == list(range(len(smiles)))


def test_murcko_scaffold_strips_side_chains():
    assert murcko_scaffold("CCCCc1ccccc1") == murcko_scaffold("CCc1ccccc1")


# ---- filters -------------------------------------------------------------
def test_reactive_group_detection():
    assert any("acyl_halide" in r for r in reactive_groups(Chem.MolFromSmiles("CC(=O)Cl")))
    assert any("epoxide" in r for r in reactive_groups(Chem.MolFromSmiles("C1OC1")))
    assert reactive_groups(Chem.MolFromSmiles("c1ccccc1")) == []


def test_property_gate_rejects_oversized_and_greasy():
    cfg = HitDiscoveryConfig()
    df = pd.DataFrame({"canonical_smiles": [
        "Cc1cc(C)nc(n1)N1CCN(CC1)C(=O)c1ccccc1",  # lead-like, should pass
        "C" * 60,                                  # huge and greasy
        "CC(=O)Cl",                                # tiny + reactive
    ]})
    out = apply_filters(df, cfg)
    assert out.loc[0, "filter_pass"], out.loc[0, "filter_reasons"]
    assert not out.loc[1, "filter_pass"]
    assert not out.loc[2, "filter_pass"]
    assert out.loc[1, "filter_reasons"]           # reason is always recorded


def test_lead_like_gate_rejects_fragments():
    """Paracetamol (151 Da) is a fragment by lead-like standards. The default
    gate is deliberately tighter than Rule-of-5, so this must fail -- if it
    starts passing, the MW floor has been loosened by accident."""
    out = apply_filters(
        pd.DataFrame({"canonical_smiles": ["CC(=O)Nc1ccc(O)cc1"]}), HitDiscoveryConfig()
    )
    assert not out.loc[0, "filter_pass"]
    assert "mw:" in out.loc[0, "filter_reasons"]


def test_filters_never_drop_rows():
    cfg = HitDiscoveryConfig()
    df = pd.DataFrame({"canonical_smiles": ["CCO", "junk", "c1ccccc1"]})
    assert len(apply_filters(df, cfg)) == 3


# ---- synthetic accessibility --------------------------------------------
def test_sa_score_ranks_simple_below_complex():
    simple = sa_score_smiles("c1ccccc1")
    complex_ = sa_score_smiles("C1CC2CCC3(C1)C(=O)N(C2)C3C1CCC2(CC1)OCCO2")
    assert 1.0 <= simple <= 10.0
    assert complex_ > simple


def test_sa_score_handles_bad_input():
    assert sa_score_smiles("nonsense") == 10.0
