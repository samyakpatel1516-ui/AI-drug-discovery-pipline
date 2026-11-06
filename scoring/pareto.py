"""Pareto analysis and diverse selection.

An MPO score collapses a trade-off into one number, which is convenient and
hides information. The Pareto front keeps the trade-off visible: these are the
compounds for which nothing else is better on every axis at once. Showing a
project team the front, rather than a single ranked list, is usually what
turns a model output into a decision.

Selection then adds the two constraints a single ranked list ignores:
structural diversity (don't buy twenty analogs of one compound) and a budget
split between exploitation and exploration.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import DataStructs

LOG = logging.getLogger("dd_pipeline.scoring.pareto")


def pareto_front(objectives: np.ndarray, maximize: list[bool] | None = None) -> np.ndarray:
    """Boolean mask of non-dominated rows. objectives: (n_samples, n_objectives)."""
    X = np.asarray(objectives, dtype=float).copy()
    if maximize is not None:
        for j, mx in enumerate(maximize):
            if not mx:
                X[:, j] = -X[:, j]
    n = len(X)
    nd = np.ones(n, dtype=bool)
    # Sort by first objective descending so most comparisons terminate early.
    order = np.argsort(-X[:, 0])
    for i in order:
        if not nd[i]:
            continue
        # j dominates i if it is >= on all objectives and > on at least one.
        dominates = np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)
        if dominates.any():
            nd[i] = False
    return nd


def pareto_ranks(objectives: np.ndarray, maximize: list[bool] | None = None,
                 max_ranks: int = 10) -> np.ndarray:
    """Non-dominated sorting: rank 1 = front, 2 = front of the remainder, ..."""
    X = np.asarray(objectives, dtype=float)
    n = len(X)
    ranks = np.full(n, max_ranks + 1, dtype=int)
    remaining = np.arange(n)
    rank = 1
    while remaining.size and rank <= max_ranks:
        mask = pareto_front(X[remaining], maximize)
        ranks[remaining[mask]] = rank
        remaining = remaining[~mask]
        rank += 1
    return ranks


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance: prefers points in sparse regions of the
    front, so a selection spans the trade-off instead of clumping."""
    X = np.asarray(objectives, dtype=float)
    n, m = X.shape
    dist = np.zeros(n)
    for j in range(m):
        order = np.argsort(X[:, j])
        col = X[order, j]
        dist[order[0]] = dist[order[-1]] = np.inf
        span = col[-1] - col[0]
        if span <= 0:
            continue
        dist[order[1:-1]] += (col[2:] - col[:-2]) / span
    return dist


def diverse_pick(
    fingerprints: list,
    scores: np.ndarray,
    n: int,
    max_similarity: float = 0.7,
) -> np.ndarray:
    """Greedy score-ordered pick with a Tanimoto similarity veto (sphere
    exclusion). Returns selected indices.

    Greedy rather than MaxMin because the score ordering should lead; the
    similarity cap is a constraint, not the objective.
    """
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    sel_fps: list = []
    for idx in order:
        fp = fingerprints[idx]
        if fp is None:
            continue
        if sel_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, sel_fps)
            if max(sims) > max_similarity:
                continue
        selected.append(int(idx))
        sel_fps.append(fp)
        if len(selected) >= n:
            break
    if len(selected) < n:
        LOG.warning(
            "diversity cap %.2f yielded only %d/%d compounds; relax it or widen the pool",
            max_similarity, len(selected), n,
        )
    return np.array(selected, dtype=int)


def select_batch(
    df: pd.DataFrame,
    fingerprints: list,
    n_total: int,
    score_col: str = "mpo_score",
    uncertainty_col: str | None = "pred_std",
    exploration_frac: float = 0.25,
    max_similarity: float = 0.7,
    in_domain_col: str | None = "in_domain",
) -> pd.DataFrame:
    """Compose a screening/synthesis batch.

    Two sub-budgets:
      * exploitation -- highest scoring, in-domain compounds. These are the
        ones you expect to work.
      * exploration -- high model uncertainty but still in-domain. These are
        the ones that teach the model the most per assay well, which is what
        makes the next iteration better rather than just confirming this one.

    This is a simple, defensible active-learning policy. Upper-confidence-bound
    selection is a one-line change to the exploration ranking if you want it.
    """
    work = df.reset_index(drop=True).copy()
    fps = list(fingerprints)

    pool = np.ones(len(work), dtype=bool)
    if in_domain_col and in_domain_col in work.columns:
        pool &= work[in_domain_col].to_numpy(dtype=bool)
        if pool.sum() < n_total:
            LOG.warning("only %d in-domain compounds for a batch of %d", int(pool.sum()), n_total)
    pool_idx = np.where(pool)[0]
    if pool_idx.size == 0:
        return work.iloc[[]].assign(selection_reason=[])

    n_explore = int(round(exploration_frac * n_total))
    n_exploit = n_total - n_explore

    scores = work[score_col].to_numpy(dtype=float)
    exploit_scores = np.full(len(work), -np.inf)
    exploit_scores[pool_idx] = scores[pool_idx]
    exploit = diverse_pick(fps, exploit_scores, n_exploit, max_similarity)

    chosen = set(exploit.tolist())
    explore = np.array([], dtype=int)
    if n_explore > 0 and uncertainty_col and uncertainty_col in work.columns:
        unc = work[uncertainty_col].to_numpy(dtype=float)
        # Rank exploration candidates by uncertainty, but keep a score floor
        # so we are not spending the budget on compounds nobody would make.
        floor = np.nanpercentile(scores[pool_idx], 40) if pool_idx.size > 2 else -np.inf
        ex_scores = np.full(len(work), -np.inf)
        ok = pool.copy()
        ok &= scores >= floor
        for i in chosen:
            ok[i] = False
        ex_scores[ok] = unc[ok]
        explore = diverse_pick(fps, ex_scores, n_explore, max_similarity)

    reason = (["exploit"] * len(exploit)) + (["explore"] * len(explore))
    picked = np.concatenate([exploit, explore]).astype(int)
    out = work.iloc[picked].copy()
    out["selection_reason"] = reason
    out["selection_rank"] = np.arange(1, len(out) + 1)
    LOG.info("batch: %d exploit + %d explore = %d compounds",
             len(exploit), len(explore), len(out))
    return out.reset_index(drop=True)
