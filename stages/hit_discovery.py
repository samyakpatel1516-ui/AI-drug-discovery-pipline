"""Stage 2: QSAR model training and virtual screening.

Order of operations is deliberate:
  standardise -> split -> featurize -> train -> evaluate -> filter -> screen -> select

Featurizing before splitting is harmless here (the featurizer is stateless),
but the applicability domain and any scaling are fit on train only. Filtering
before screening saves most of the inference cost and keeps garbage out of the
top of the ranking.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..chem.featurize import Featurizer
from ..chem.filters import apply_filters
from ..chem.splits import make_split, murcko_scaffold
from ..chem.synth import sa_score_smiles
from ..config import PipelineConfig
from ..models.applicability import ApplicabilityDomain
from ..models.evaluate import baseline_metrics, evaluate_regression
from ..models.qsar import EnsembleQSAR
from ..registry import Run
from ..scoring.pareto import select_batch

LOG = logging.getLogger("dd_pipeline.stages.hit_discovery")


def train_activity_model(
    assay: pd.DataFrame,
    cfg: PipelineConfig,
    run: Run,
    endpoint: str = "pIC50",
    smiles_col: str = "canonical_smiles",
) -> tuple[EnsembleQSAR, Featurizer, ApplicabilityDomain, dict]:
    """Train and honestly evaluate the activity model."""
    LOG.info("--- training activity model on %s (%d compounds) ---", endpoint, len(assay))

    data = assay.dropna(subset=[endpoint]).reset_index(drop=True)
    featurizer = Featurizer(cfg.featurizer)

    tr_idx, va_idx, te_idx = make_split(data, cfg.split, smiles_col=smiles_col)
    if len(va_idx) < 20:
        # Conformal calibration needs a calibration set; borrow from train
        # rather than silently skipping calibration.
        LOG.warning("validation fold too small (%d); reallocating from train", len(va_idx))
        n_take = min(max(20 - len(va_idx), 0), max(len(tr_idx) - 50, 0))
        va_idx = np.concatenate([va_idx, tr_idx[:n_take]])
        tr_idx = tr_idx[n_take:]

    fm = featurizer.transform(data[smiles_col].tolist())
    X, y = fm.X, data[endpoint].to_numpy(dtype=float)

    # Weight by replicate count: a compound measured three times is better
    # evidence than one measured once.
    w = None
    if "n_replicates" in data.columns:
        w = np.sqrt(data["n_replicates"].fillna(1).to_numpy(dtype=float))

    model = EnsembleQSAR(cfg.activity_model, endpoint=endpoint)
    model.fit(
        X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
        feature_names=fm.feature_names,
        sample_weight=w[tr_idx] if w is not None else None,
    )

    ad = ApplicabilityDomain(cfg.applicability, featurizer)
    n_fp = cfg.featurizer.fp_bits if cfg.featurizer.kind != "descriptors" else 0
    ad.fit(data.loc[tr_idx, smiles_col].tolist(), X[tr_idx][:, n_fp:] if n_fp else X[tr_idx])

    # --- evaluation ------------------------------------------------------
    spread_col = f"{endpoint}_spread"
    assay_noise = None
    if spread_col in data.columns:
        reps = data[data.get("n_replicates", pd.Series(1, index=data.index)) > 1]
        if len(reps) > 5:
            # Range/1.128 estimates sigma for n=2 replicates.
            assay_noise = float(reps[spread_col].median() / 1.128)
            LOG.info("estimated assay noise (sigma) from replicates: %.3f", assay_noise)

    active_threshold = cfg.hit_discovery.min_predicted_activity
    pred_te = model.predict(X[te_idx])
    metrics = evaluate_regression(y[te_idx], pred_te, active_threshold, assay_noise)

    fps = featurizer.bit_fingerprints(data[smiles_col].tolist())
    metrics.update(baseline_metrics(
        y[tr_idx], y[te_idx],
        [fps[i] for i in tr_idx], [fps[i] for i in te_idx], task="regression",
    ))
    # Validation metrics too -- a large train/valid/test gap is the signature
    # of either leakage or a split that is harder than you thought.
    pred_va = model.predict(X[va_idx])
    metrics["valid_rmse"] = float(np.sqrt(np.mean((y[va_idx] - pred_va.mean) ** 2)))
    metrics["train_scaffolds"] = int(
        pd.Series([murcko_scaffold(s) for s in data.loc[tr_idx, smiles_col]]).nunique()
    )
    metrics["test_scaffolds"] = int(
        pd.Series([murcko_scaffold(s) for s in data.loc[te_idx, smiles_col]]).nunique()
    )

    model.card.n_test = int(len(te_idx))
    model.card.split_strategy = cfg.split.strategy
    model.card.featurizer = cfg.featurizer.model_dump()
    model.card.metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    model.card.assay_noise_estimate = assay_noise
    model.card.notes = (
        "Trained inside dd_pipeline. Evaluated on a held-out "
        f"{cfg.split.strategy} split; predictions outside the applicability "
        "domain are flagged and must not be used for purchase decisions."
    )
    model.save(run.path("models", f"activity_{endpoint}.pkl"))

    run.log_metrics(f"activity_model_{endpoint}", metrics)
    LOG.info(
        "test: RMSE=%.3f  Spearman=%.3f  EF@1%%=%.1f  (1-NN baseline RMSE=%.3f)",
        metrics["rmse"], metrics["spearman"],
        metrics.get("ef_1pct", float("nan")), metrics.get("baseline_1nn_rmse", float("nan")),
    )
    imp = model.feature_importance(20)
    run.save_json({"top_features": imp, "metrics": metrics},
                  f"activity_model_{endpoint}_report", "hit_discovery")
    return model, featurizer, ad, metrics


def virtual_screen(
    library: pd.DataFrame,
    model: EnsembleQSAR,
    featurizer: Featurizer,
    ad: ApplicabilityDomain,
    cfg: PipelineConfig,
    run: Run,
    smiles_col: str = "canonical_smiles",
    batch_size: int = 20_000,
    known_smiles: set[str] | None = None,
) -> pd.DataFrame:
    """Score a library, flag domain, and pick a diverse, budgeted hit list."""
    LOG.info("--- virtual screening %d compounds ---", len(library))

    lib = apply_filters(library, cfg.hit_discovery, smiles_col=smiles_col)
    screenable = lib[lib["filter_pass"]].reset_index(drop=True)
    if screenable.empty:
        raise RuntimeError("every library compound failed the filters; loosen hit_discovery gates")

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    desc_blocks: list[np.ndarray] = []
    n_fp = cfg.featurizer.fp_bits if cfg.featurizer.kind != "descriptors" else 0

    for start in range(0, len(screenable), batch_size):
        chunk = screenable.iloc[start:start + batch_size]
        fm = featurizer.transform(chunk[smiles_col].tolist())
        pred = model.predict(fm.X)
        means.append(pred.mean)
        stds.append(pred.std)
        desc_blocks.append(fm.X[:, n_fp:] if n_fp else fm.X)
        LOG.info("scored %d/%d", min(start + batch_size, len(screenable)), len(screenable))

    out = screenable.copy()
    out["pred_activity"] = np.concatenate(means)
    out["pred_std"] = np.concatenate(stds)

    dom = ad.assess(out[smiles_col].tolist(), np.vstack(desc_blocks))
    for k, v in dom.items():
        out[k] = v

    out["sa_score"] = [sa_score_smiles(s) for s in out[smiles_col]]
    # Lower confidence bound: rank by what we can defend, not by the mean.
    out["pred_lcb"] = out["pred_activity"] - 1.0 * out["pred_std"]

    if known_smiles:
        out["is_known"] = out[smiles_col].isin(known_smiles)
        n_known = int(out["is_known"].sum())
        if n_known:
            LOG.info("%d screened compounds are already in the training set", n_known)
    else:
        out["is_known"] = False

    run.save_table(out, "virtual_screen_scored", "hit_discovery")

    # --- hit selection ---------------------------------------------------
    pool = out[
        (out["pred_activity"] >= cfg.hit_discovery.min_predicted_activity)
        & (~out["is_known"])
    ].reset_index(drop=True)
    LOG.info("%d compounds above the %.1f activity threshold",
             len(pool), cfg.hit_discovery.min_predicted_activity)
    if pool.empty:
        LOG.warning("no compound cleared the activity threshold; falling back to top 5%%")
        pool = out.nlargest(max(int(0.05 * len(out)), cfg.hit_discovery.n_hits), "pred_activity")
        pool = pool.reset_index(drop=True)

    pool = pool.copy()
    pool["mpo_score"] = pool["pred_lcb"]  # hit stage ranks on defensible potency
    fps = featurizer.bit_fingerprints(pool[smiles_col].tolist())
    hits = select_batch(
        pool, fps, n_total=cfg.hit_discovery.n_hits,
        score_col="mpo_score", uncertainty_col="pred_std",
        exploration_frac=cfg.hit_discovery.exploration_frac,
        max_similarity=cfg.hit_discovery.max_pairwise_similarity,
        in_domain_col="in_domain",
    )
    hits["scaffold"] = [murcko_scaffold(s) for s in hits[smiles_col]]
    run.save_table(hits, "hit_list", "hit_discovery")

    run.log_metrics("hit_discovery", {
        "n_library": int(len(library)),
        "n_passed_filters": int(len(screenable)),
        "filter_pass_rate": float(len(screenable) / max(len(library), 1)),
        "n_above_threshold": int(len(pool)),
        "n_hits_selected": int(len(hits)),
        "n_unique_scaffolds_in_hits": int(hits["scaffold"].nunique()),
        "frac_hits_in_domain": float(hits["in_domain"].mean()) if len(hits) else 0.0,
        "mean_hit_pred_activity": float(hits["pred_activity"].mean()) if len(hits) else None,
        "mean_hit_sa_score": float(hits["sa_score"].mean()) if len(hits) else None,
    })
    return hits
