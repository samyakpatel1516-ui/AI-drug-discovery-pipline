"""Iterative design-make-test-analyse cycles.

A discovery pipeline is not a one-shot prediction; it is a loop. You screen a
batch, the assay comes back, you retrain, and you choose the next batch. The
thing worth measuring is therefore not "how good is the model" but "how fast
does the programme improve per assay plate" -- and that depends almost entirely
on how the batch is chosen, which is what this module exercises.

`oracle_fn` is the seam. Here it is the simulator. In production it is the
function that reads this week's assay results out of your LIMS: same loop, same
selection policy, same metrics.

The comparison that matters is against the greedy policy and against random.
Beating random is the minimum bar -- if the model-driven policies do not clear
it, the model is not earning its keep. Beating *greedy* is a harder and more
situational claim: on a smooth landscape with a large pool, greedy often ties
with uncertainty-aware selection, and the diversity cap mainly buys you
insurance against a series that turns out to be a dead end. Run
`compare_policies` on your own data rather than assuming; the answer depends
on your landscape, not on which policy is fashionable.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from ..chem.featurize import Featurizer
from ..chem.splits import murcko_scaffold
from ..config import PipelineConfig
from ..models.applicability import ApplicabilityDomain
from ..models.qsar import EnsembleQSAR
from ..registry import Run
from ..scoring.pareto import diverse_pick, select_batch

LOG = logging.getLogger("dd_pipeline.stages.active_learning")

# A policy takes (candidate table, fingerprints, batch size) -> chosen indices.
Policy = Callable[[pd.DataFrame, list, int], np.ndarray]


def greedy_policy(df: pd.DataFrame, fps: list, n: int) -> np.ndarray:
    """Pure exploitation: take the top predictions. The baseline to beat."""
    return np.argsort(df["pred"].to_numpy(dtype=float))[::-1][:n]


def random_policy(df: pd.DataFrame, fps: list, n: int, seed: int = 0) -> np.ndarray:
    """No model at all. If the clever policies do not beat this, stop."""
    return np.random.default_rng(seed).choice(len(df), size=min(n, len(df)), replace=False)


def ucb_policy(df: pd.DataFrame, fps: list, n: int, beta: float = 1.5) -> np.ndarray:
    """Upper confidence bound: mean + beta * sigma.

    One knob that trades off exploitation and exploration continuously. beta=0
    is greedy; large beta chases uncertainty.
    """
    score = df["pred"].to_numpy(dtype=float) + beta * df["pred_std"].to_numpy(dtype=float)
    return np.argsort(score)[::-1][:n]


def diverse_ucb_policy(df: pd.DataFrame, fps: list, n: int, beta: float = 1.5,
                       max_similarity: float = 0.65) -> np.ndarray:
    """UCB with a structural diversity cap -- the policy this pipeline defaults
    to. A batch of near-identical compounds wastes most of its wells, because
    the second analog tells you almost nothing the first did not."""
    score = df["pred"].to_numpy(dtype=float) + beta * df["pred_std"].to_numpy(dtype=float)
    return diverse_pick(fps, score, n, max_similarity=max_similarity)


def explore_exploit_policy(df: pd.DataFrame, fps: list, n: int,
                           exploration_frac: float = 0.3,
                           max_similarity: float = 0.65) -> np.ndarray:
    """Explicit split budget, using the same selector as the main pipeline."""
    work = df.copy()
    work["mpo_score"] = work["pred"]
    picked = select_batch(
        work, fps, n_total=n, score_col="mpo_score", uncertainty_col="pred_std",
        exploration_frac=exploration_frac, max_similarity=max_similarity,
        in_domain_col="in_domain" if "in_domain" in work.columns else None,
    )
    # select_batch returns rows; map back to positional indices.
    key = work["canonical_smiles"] if "canonical_smiles" in work else work["smiles"]
    lookup = {s: i for i, s in enumerate(key)}
    pkey = picked["canonical_smiles"] if "canonical_smiles" in picked else picked["smiles"]
    return np.array([lookup[s] for s in pkey if s in lookup], dtype=int)


POLICIES: dict[str, Policy] = {
    "random": random_policy,
    "greedy": greedy_policy,
    "ucb": ucb_policy,
    "diverse_ucb": diverse_ucb_policy,
    "explore_exploit": explore_exploit_policy,
}


def run_active_learning(
    pool: pd.DataFrame,
    oracle_fn: Callable[[list[str]], np.ndarray],
    cfg: PipelineConfig,
    run: Run | None = None,
    initial: pd.DataFrame | None = None,
    n_rounds: int = 5,
    batch_size: int = 50,
    policy: str | Policy = "diverse_ucb",
    endpoint: str = "pIC50",
    smiles_col: str = "canonical_smiles",
    n_initial: int = 100,
) -> dict[str, pd.DataFrame]:
    """Run `n_rounds` of select -> measure -> retrain.

    pool      -- unmeasured compounds available to select from
    oracle_fn -- SMILES list -> measured values (your assay, or the simulator)
    initial   -- already-measured starting data; sampled from the pool if None

    Returns the per-round history and the accumulated training set.
    """
    policy_fn = POLICIES[policy] if isinstance(policy, str) else policy
    policy_name = policy if isinstance(policy, str) else getattr(policy, "__name__", "custom")
    featurizer = Featurizer(cfg.featurizer)
    rng = np.random.default_rng(cfg.seed)

    work = pool.reset_index(drop=True).copy()
    if smiles_col not in work.columns:
        raise ValueError(f"pool must contain a {smiles_col!r} column")

    # --- seed the training set -------------------------------------------
    if initial is None:
        start = rng.choice(len(work), size=min(n_initial, len(work)), replace=False)
        train = work.iloc[start].copy()
        train[endpoint] = oracle_fn(train[smiles_col].tolist())
        work = work.drop(index=work.index[start]).reset_index(drop=True)
    else:
        train = initial.reset_index(drop=True).copy()

    LOG.info("active learning [%s]: %d measured to start, %d in the pool, "
             "%d rounds x %d compounds", policy_name, len(train), len(work),
             n_rounds, batch_size)

    history: list[dict] = []
    for rnd in range(1, n_rounds + 1):
        if work.empty:
            LOG.info("pool exhausted at round %d", rnd)
            break

        # --- retrain on everything measured so far -----------------------
        fm_tr = featurizer.transform(train[smiles_col].tolist())
        y_tr = train[endpoint].to_numpy(dtype=float)
        n_hold = max(int(0.15 * len(train)), 20)
        perm = rng.permutation(len(train))
        cal_idx, fit_idx = perm[:n_hold], perm[n_hold:]

        model = EnsembleQSAR(cfg.activity_model, endpoint=endpoint)
        model.fit(fm_tr.X[fit_idx], y_tr[fit_idx], fm_tr.X[cal_idx], y_tr[cal_idx],
                  feature_names=fm_tr.feature_names)

        ad = ApplicabilityDomain(cfg.applicability, featurizer)
        n_fp = cfg.featurizer.fp_bits if cfg.featurizer.kind != "descriptors" else 0
        ad.fit(train[smiles_col].tolist(),
               fm_tr.X[:, n_fp:] if n_fp else fm_tr.X)

        # --- score the pool ----------------------------------------------
        fm_pool = featurizer.transform(work[smiles_col].tolist())
        pred = model.predict(fm_pool.X)
        cand = work.copy()
        cand["pred"] = pred.mean
        cand["pred_std"] = pred.std
        dom = ad.assess(work[smiles_col].tolist(),
                        fm_pool.X[:, n_fp:] if n_fp else fm_pool.X)
        cand["in_domain"] = dom["in_domain"]

        fps = featurizer.bit_fingerprints(cand[smiles_col].tolist())

        # --- select and "measure" ----------------------------------------
        pick = policy_fn(cand, fps, batch_size)
        pick = np.asarray(pick, dtype=int)
        if pick.size == 0:
            LOG.warning("policy selected nothing at round %d; stopping", rnd)
            break
        batch = cand.iloc[pick].copy()
        measured = oracle_fn(batch[smiles_col].tolist())
        batch[endpoint] = measured
        batch["round"] = rnd

        # --- what did the round buy us? ----------------------------------
        rec = {
            "round": rnd,
            "policy": policy_name,
            "n_train_before": int(len(train)),
            "n_selected": int(len(batch)),
            "batch_mean_measured": float(np.mean(measured)),
            "batch_max_measured": float(np.max(measured)),
            "batch_pred_vs_measured_rmse": float(
                np.sqrt(np.mean((batch["pred"].to_numpy(dtype=float) - measured) ** 2))
            ),
            "batch_hit_rate": float(np.mean(measured >= cfg.hit_discovery.min_predicted_activity)),
            "batch_unique_scaffolds": int(
                pd.Series([murcko_scaffold(s) for s in batch[smiles_col]]).nunique()
            ),
            "batch_frac_in_domain": float(batch["in_domain"].mean()),
            # Cumulative best is the number a programme actually cares about:
            # how good is the best compound we have found so far.
            "cumulative_best": float(max(train[endpoint].max(), np.max(measured))),
        }
        if len(batch) > 3:
            from scipy import stats as _st
            rec["batch_pred_vs_measured_spearman"] = float(
                _st.spearmanr(batch["pred"].to_numpy(dtype=float), measured).statistic
            )
        history.append(rec)
        LOG.info(
            "round %d [%s]: mean=%.2f max=%.2f hit_rate=%.2f scaffolds=%d "
            "prospective_rmse=%.3f cumulative_best=%.2f",
            rnd, policy_name, rec["batch_mean_measured"], rec["batch_max_measured"],
            rec["batch_hit_rate"], rec["batch_unique_scaffolds"],
            rec["batch_pred_vs_measured_rmse"], rec["cumulative_best"],
        )

        train = pd.concat([train, batch.drop(columns=["pred", "pred_std", "in_domain"],
                                             errors="ignore")], ignore_index=True)
        work = work.drop(index=work.index[pick]).reset_index(drop=True)

    hist = pd.DataFrame(history)
    if run is not None:
        run.save_table(hist, f"active_learning_{policy_name}", "active_learning")
        run.save_table(train, f"active_learning_train_{policy_name}", "active_learning")
        if len(hist):
            run.log_metrics(f"active_learning_{policy_name}", {
                "n_rounds": int(len(hist)),
                "final_cumulative_best": float(hist["cumulative_best"].iloc[-1]),
                "final_batch_hit_rate": float(hist["batch_hit_rate"].iloc[-1]),
                "mean_prospective_rmse": float(hist["batch_pred_vs_measured_rmse"].mean()),
                "total_measured": int(len(train)),
            })
    return {"history": hist, "train": train}


def compare_policies(
    pool: pd.DataFrame,
    oracle_fn: Callable[[list[str]], np.ndarray],
    cfg: PipelineConfig,
    run: Run | None = None,
    policies: tuple[str, ...] = ("random", "greedy", "diverse_ucb"),
    n_rounds: int = 4,
    batch_size: int = 50,
    **kwargs,
) -> pd.DataFrame:
    """Run the same loop under several policies and tabulate the outcome.

    Each policy starts from the same seed data and the same pool, so the
    comparison is of selection strategy alone. This is the experiment worth
    running before committing a real budget to a screening cadence.
    """
    frames = []
    for pol in policies:
        LOG.info("=== policy: %s ===", pol)
        res = run_active_learning(
            pool.copy(), oracle_fn, cfg, run=run,
            n_rounds=n_rounds, batch_size=batch_size, policy=pol, **kwargs,
        )
        frames.append(res["history"])
    out = pd.concat(frames, ignore_index=True)
    if run is not None:
        run.save_table(out, "active_learning_policy_comparison", "active_learning")
    summary = out.groupby("policy").agg(
        final_best=("cumulative_best", "max"),
        mean_hit_rate=("batch_hit_rate", "mean"),
        mean_prospective_rmse=("batch_pred_vs_measured_rmse", "mean"),
    )
    LOG.info("policy comparison:\n%s", summary.to_string())
    return out
