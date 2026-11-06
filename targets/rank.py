"""Target prioritisation: combine evidence into a ranked, explainable shortlist.

The output is deliberately not just a score. Each target carries the component
scores that produced it and a short text rationale, because the conversation
that follows is always "why is this one above that one?" -- and a ranking
nobody can interrogate does not get acted on.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import TargetIDConfig
from .druggability import recommend_modality

LOG = logging.getLogger("dd_pipeline.targets.rank")


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    if hi - lo < 1e-12:
        return np.where(finite, 0.5, 0.0)
    out = np.clip((x - lo) / (hi - lo), 0, 1)
    return np.where(finite, out, 0.0)


def build_target_table(
    de: pd.DataFrame,
    dependency: pd.DataFrame,
    annotations: pd.DataFrame,
    specificity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join the evidence layers on `gene`, keeping genes with any evidence."""
    tbl = de.merge(dependency, on="gene", how="outer")
    tbl = tbl.merge(annotations, on="gene", how="left")
    if specificity is not None:
        tbl = tbl.merge(specificity, on="gene", how="left")
    LOG.info("target evidence table: %d genes x %d columns", len(tbl), tbl.shape[1])
    return tbl


def score_targets(tbl: pd.DataFrame, cfg: TargetIDConfig | None = None) -> pd.DataFrame:
    """Weighted composite over five normalised evidence components."""
    cfg = cfg or TargetIDConfig()
    out = tbl.copy()
    n = len(out)

    # --- differential expression: effect size, gated by significance -------
    if "log2fc" in out and "fdr" in out:
        lfc = out["log2fc"].to_numpy(dtype=float)
        fdr = out["fdr"].to_numpy(dtype=float)
        sig = (np.abs(lfc) >= cfg.min_abs_log2fc) & (fdr <= cfg.max_fdr)
        # Only up-regulation counts for an inhibitor programme; flip the sign
        # (or take |lfc|) if you are after an activator or a synthetic-lethal.
        de_component = _minmax(np.where(sig, np.clip(lfc, 0, None), 0.0))
    else:
        de_component = np.zeros(n)
        LOG.warning("no differential expression columns; DE component is zero")
    out["score_differential_expression"] = de_component

    # --- dependency: strength x selectivity -------------------------------
    if "dep_mean_disease" in out:
        strength = _minmax(-out["dep_mean_disease"].to_numpy(dtype=float))
        sel = _minmax(out.get("dep_selectivity", pd.Series(np.zeros(n))).to_numpy(dtype=float))
        dep_component = 0.5 * strength + 0.5 * sel
        if "is_common_essential" in out:
            # A pan-essential gene is a chemotherapy mechanism at best.
            penalty = out["is_common_essential"].fillna(False).astype(float).to_numpy()
            dep_component = dep_component / (1.0 + (cfg.normal_essentiality_penalty - 1.0) * penalty)
    else:
        dep_component = np.zeros(n)
    out["score_dependency"] = np.clip(dep_component, 0, 1)

    out["score_druggability"] = np.clip(
        out.get("druggability", pd.Series(np.zeros(n))).fillna(0).to_numpy(dtype=float), 0, 1
    )
    out["score_safety"] = np.clip(
        out.get("safety", pd.Series(np.full(n, 0.5))).fillna(0.5).to_numpy(dtype=float), 0, 1
    )
    out["score_tractability_evidence"] = np.clip(
        out.get("ligand_evidence", pd.Series(np.zeros(n))).fillna(0).to_numpy(dtype=float), 0, 1
    )
    if "specificity" in out:
        # Tissue specificity folds into safety rather than standing alone.
        spec = np.clip(out["specificity"].fillna(0.5).to_numpy(dtype=float), 0, 1)
        out["score_safety"] = np.clip(0.7 * out["score_safety"] + 0.3 * spec, 0, 1)

    total = np.zeros(n)
    for key, w in cfg.weights.items():
        col = f"score_{key}"
        if col not in out:
            LOG.warning("weight %r has no matching score column; ignored", key)
            continue
        total += w * out[col].to_numpy(dtype=float)
    out["target_score"] = total

    out["recommended_modality"] = out.apply(recommend_modality, axis=1)
    out["rationale"] = [_rationale(r, cfg) for _, r in out.iterrows()]
    out = out.sort_values("target_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def _rationale(row: pd.Series, cfg: TargetIDConfig) -> str:
    """One readable sentence per target. This is what gets pasted into a deck."""
    bits: list[str] = []
    lfc, fdr = row.get("log2fc"), row.get("fdr")
    if pd.notna(lfc) and pd.notna(fdr) and abs(lfc) >= cfg.min_abs_log2fc and fdr <= cfg.max_fdr:
        bits.append(f"{lfc:+.1f} log2FC (FDR {fdr:.1e})")
    dep = row.get("dep_mean_disease")
    if pd.notna(dep) and dep <= cfg.dependency_threshold:
        sel = row.get("dep_selectivity")
        tail = f", selectivity {sel:+.2f}" if pd.notna(sel) else ""
        bits.append(f"dependency {dep:.2f}{tail}")
    if row.get("is_common_essential", False):
        bits.append("pan-essential (toxicity risk)")
    drugg = row.get("druggability")
    if pd.notna(drugg):
        src = row.get("druggability_source", "")
        bits.append(f"druggability {drugg:.2f}" + (f" ({src})" if src else ""))
    if row.get("has_approved_drug", False):
        bits.append("approved drug precedent")
    elif row.get("n_known_ligands", 0) and row.get("n_known_ligands", 0) > 0:
        bits.append(f"{int(row['n_known_ligands'])} known ligands")
    safety = row.get("safety")
    if pd.notna(safety) and safety < 0.5:
        bits.append(f"safety concern ({safety:.2f})")
    return "; ".join(bits) if bits else "insufficient evidence"


def shortlist(scored: pd.DataFrame, n: int = 20,
              min_druggability: float = 0.4,
              exclude_common_essential: bool = True) -> pd.DataFrame:
    """Apply the hard gates a portfolio committee would apply anyway."""
    df = scored.copy()
    before = len(df)
    if exclude_common_essential and "is_common_essential" in df:
        df = df[~df["is_common_essential"].fillna(False).astype(bool)]
    if "druggability" in df:
        df = df[df["druggability"].fillna(0) >= min_druggability]
    LOG.info("shortlist gates: %d -> %d genes, taking top %d", before, len(df), n)
    return df.head(n).reset_index(drop=True)
