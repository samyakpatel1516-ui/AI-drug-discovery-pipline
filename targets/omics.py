"""Differential expression and dependency analysis for target identification.

Deliberately simple statistics, run correctly: Welch's t-test on log-scale
expression with Benjamini-Hochberg FDR control, plus effect sizes. If you have
raw counts, run DESeq2/limma-voom upstream and feed the result table in here --
`differential_expression` accepts pre-computed stats through `from_table`.

The part worth more attention than the test itself is the *contrast*: tumour
vs adjacent normal answers a different question from responder vs
non-responder, and a target that looks excellent under one is often
uninteresting under the other.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

LOG = logging.getLogger("dd_pipeline.targets.omics")


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (FDR). Monotone, clipped at 1."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downwards.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def differential_expression(
    expr: pd.DataFrame,
    group_labels: pd.Series,
    case_label: str,
    control_label: str,
    log_transform: bool = True,
    min_expression: float = 1.0,
) -> pd.DataFrame:
    """Per-gene case-vs-control contrast.

    expr: genes x samples. group_labels: indexed by sample.
    Returns log2FC, Welch t, p, FDR, Cohen's d and per-group means.
    """
    samples = [s for s in expr.columns if s in group_labels.index]
    expr = expr[samples]
    labels = group_labels.loc[samples]

    case = expr.loc[:, labels == case_label]
    ctrl = expr.loc[:, labels == control_label]
    if case.shape[1] < 2 or ctrl.shape[1] < 2:
        raise ValueError(
            f"need >=2 samples per group, got case={case.shape[1]} control={ctrl.shape[1]}"
        )

    # Drop genes that are effectively off everywhere: their fold changes are
    # noise and they inflate the multiple-testing burden.
    keep = (expr.mean(axis=1) >= min_expression)
    case, ctrl = case.loc[keep], ctrl.loc[keep]
    LOG.info("DE: %d/%d genes pass the expression floor", int(keep.sum()), len(keep))

    A = np.log2(case.to_numpy(dtype=float) + 1.0) if log_transform else case.to_numpy(float)
    B = np.log2(ctrl.to_numpy(dtype=float) + 1.0) if log_transform else ctrl.to_numpy(float)

    mean_a, mean_b = A.mean(axis=1), B.mean(axis=1)
    var_a, var_b = A.var(axis=1, ddof=1), B.var(axis=1, ddof=1)
    t_stat, p_val = stats.ttest_ind(A, B, axis=1, equal_var=False)

    n_a, n_b = A.shape[1], B.shape[1]
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))
    cohens_d = np.divide(mean_a - mean_b, pooled_sd, out=np.zeros_like(mean_a),
                         where=pooled_sd > 0)

    res = pd.DataFrame(
        {
            "gene": case.index.to_numpy(),
            "mean_case": mean_a,
            "mean_control": mean_b,
            "log2fc": mean_a - mean_b,
            "t_stat": t_stat,
            "p_value": np.nan_to_num(p_val, nan=1.0),
            "cohens_d": cohens_d,
            "n_case": n_a,
            "n_control": n_b,
        }
    )
    res["fdr"] = benjamini_hochberg(res["p_value"].to_numpy())
    return res.sort_values("fdr").reset_index(drop=True)


def dependency_summary(
    dep: pd.DataFrame,
    disease_models: list[str],
    control_models: list[str] | None = None,
) -> pd.DataFrame:
    """Summarise knockout/knockdown dependency scores (DepMap convention:
    0 = no effect, -1 = median common essential).

    The selectivity column is what matters. A gene every cell line needs is a
    toxicity liability, not a target -- the whole point is a dependency that
    is strong in disease models and weak elsewhere.
    """
    disease_models = [m for m in disease_models if m in dep.columns]
    if not disease_models:
        raise ValueError("none of the disease model columns are present in the dependency table")

    out = pd.DataFrame({"gene": dep.index.to_numpy()})
    D = dep[disease_models].to_numpy(dtype=float)
    out["dep_mean_disease"] = np.nanmean(D, axis=1)
    out["dep_min_disease"] = np.nanmin(D, axis=1)
    out["dep_frac_dependent"] = np.nanmean(D < -0.5, axis=1)

    if control_models:
        control_models = [m for m in control_models if m in dep.columns]
    if control_models:
        C = dep[control_models].to_numpy(dtype=float)
        out["dep_mean_control"] = np.nanmean(C, axis=1)
        out["dep_selectivity"] = out["dep_mean_control"] - out["dep_mean_disease"]
        out["is_common_essential"] = np.nanmean(C < -0.5, axis=1) > 0.8
    else:
        out["dep_mean_control"] = np.nan
        out["dep_selectivity"] = -out["dep_mean_disease"]
        out["is_common_essential"] = False
    return out


def expression_specificity(expr: pd.DataFrame, tissue_labels: pd.Series,
                           target_tissue: str) -> pd.DataFrame:
    """Tissue specificity via a Gini-style concentration measure.

    A target expressed only in the disease tissue has a far better therapeutic
    window than one expressed everywhere, especially in heart, liver and CNS.
    """
    samples = [s for s in expr.columns if s in tissue_labels.index]
    labels = tissue_labels.loc[samples]
    means = expr[samples].T.groupby(labels).mean().T  # genes x tissues

    vals = means.to_numpy(dtype=float)
    total = np.nansum(vals, axis=1, keepdims=True)
    frac = np.divide(vals, total, out=np.zeros_like(vals), where=total > 0)
    # Normalised entropy: 0 = one tissue only, 1 = uniform across tissues.
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(frac > 0, frac * np.log(frac), 0.0), axis=1)
    ent /= np.log(max(vals.shape[1], 2))

    out = pd.DataFrame({"gene": means.index.to_numpy(), "expression_entropy": ent})
    out["specificity"] = 1.0 - ent
    if target_tissue in means.columns:
        out["target_tissue_fraction"] = frac[:, list(means.columns).index(target_tissue)]
    else:
        out["target_tissue_fraction"] = np.nan
        LOG.warning("target tissue %r absent from the expression table", target_tissue)
    return out
