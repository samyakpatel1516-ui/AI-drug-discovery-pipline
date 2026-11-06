"""Data splitting.

The single most common way to ship a QSAR model that works in validation and
fails in the lab is a random split over a congeneric series. Scaffold and
cluster splits force the model to generalise to chemistry it has not seen,
which is the question you actually care about when you screen a new library.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from ..config import SplitConfig

LOG = logging.getLogger("dd_pipeline.chem.splits")


def murcko_scaffold(smiles: str, generic: bool = False) -> str:
    """Bemis-Murcko scaffold SMILES. `generic` collapses atom types so that
    bioisosteric cores land in the same bucket -- a stricter, more honest split."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return ""


def scaffold_split(
    smiles: list[str], test_frac: float, valid_frac: float, seed: int = 1337,
    generic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic scaffold split, large scaffold groups first.

    Big groups go to train so that test/valid are made of many small, distinct
    series rather than one giant family.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles):
        groups[murcko_scaffold(smi, generic=generic)].append(i)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n = len(smiles)
    n_test, n_valid = int(np.floor(test_frac * n)), int(np.floor(valid_frac * n))

    train: list[int] = []
    valid: list[int] = []
    test: list[int] = []
    for grp in ordered:
        if len(test) + len(grp) <= n_test:
            test.extend(grp)
        elif len(valid) + len(grp) <= n_valid:
            valid.extend(grp)
        else:
            train.extend(grp)

    rng = np.random.default_rng(seed)
    out = []
    for idx in (train, valid, test):
        a = np.array(sorted(idx), dtype=int)
        rng.shuffle(a)
        out.append(a)
    LOG.info(
        "scaffold split: train=%d valid=%d test=%d over %d scaffolds",
        len(out[0]), len(out[1]), len(out[2]), len(groups),
    )
    return out[0], out[1], out[2]


def random_split(n: int, test_frac: float, valid_frac: float, seed: int = 1337):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test, n_valid = int(n * test_frac), int(n * valid_frac)
    return idx[n_test + n_valid:], idx[n_test:n_test + n_valid], idx[:n_test]


def temporal_split(dates: pd.Series, test_frac: float, valid_frac: float):
    """Split by measurement date -- the closest analogue of prospective use.

    Report this metric to decision makers. It is usually the worst-looking
    number and the only one that has predicted real screening performance.
    """
    order = np.argsort(pd.to_datetime(dates).values, kind="stable")
    n = len(order)
    n_test, n_valid = int(n * test_frac), int(n * valid_frac)
    n_train = n - n_test - n_valid
    return order[:n_train], order[n_train:n_train + n_valid], order[n_train + n_valid:]


def butina_cluster_split(
    smiles: list[str], test_frac: float, valid_frac: float, seed: int = 1337,
    cutoff: float = 0.4, fp_generator=None,
):
    """Cluster by Tanimoto distance then split whole clusters.

    Stricter than scaffold splitting for series that share a scaffold only
    loosely. Expensive: O(n^2) similarity, fine to a few 10^4 compounds.
    """
    from rdkit.Chem import rdFingerprintGenerator

    gen = fp_generator or rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps, keep = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fps.append(gen.GetFingerprint(mol))
            keep.append(i)

    dists: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)
    groups = [[keep[j] for j in c] for c in clusters]
    groups.sort(key=lambda g: (-len(g), g[0]))

    n = len(smiles)
    n_test, n_valid = int(n * test_frac), int(n * valid_frac)
    train, valid, test = [], [], []
    for grp in groups:
        if len(test) + len(grp) <= n_test:
            test.extend(grp)
        elif len(valid) + len(grp) <= n_valid:
            valid.extend(grp)
        else:
            train.extend(grp)
    rng = np.random.default_rng(seed)
    res = []
    for idx in (train, valid, test):
        a = np.array(sorted(idx), dtype=int)
        rng.shuffle(a)
        res.append(a)
    LOG.info("butina split: %d clusters at cutoff %.2f", len(groups), cutoff)
    return tuple(res)


def make_split(df: pd.DataFrame, cfg: SplitConfig, smiles_col: str = "canonical_smiles"):
    smiles = df[smiles_col].astype(str).tolist()
    if cfg.strategy == "scaffold":
        return scaffold_split(smiles, cfg.test_frac, cfg.valid_frac, cfg.seed)
    if cfg.strategy == "random":
        return random_split(len(df), cfg.test_frac, cfg.valid_frac, cfg.seed)
    if cfg.strategy == "cluster":
        return butina_cluster_split(smiles, cfg.test_frac, cfg.valid_frac, cfg.seed)
    if cfg.strategy == "temporal":
        if not cfg.date_column or cfg.date_column not in df.columns:
            raise ValueError("temporal split requires split.date_column present in the data")
        return temporal_split(df[cfg.date_column], cfg.test_frac, cfg.valid_frac)
    raise ValueError(f"unknown split strategy {cfg.strategy}")
