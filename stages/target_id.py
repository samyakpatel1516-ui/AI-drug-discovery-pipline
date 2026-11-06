"""Stage 1: target identification and prioritisation."""
from __future__ import annotations

import logging

import pandas as pd

from ..config import PipelineConfig
from ..registry import Run
from ..targets.druggability import assess_druggability
from ..targets.omics import (
    dependency_summary,
    differential_expression,
    expression_specificity,
)
from ..targets.rank import build_target_table, score_targets, shortlist

LOG = logging.getLogger("dd_pipeline.stages.target_id")


def run_target_identification(
    expression: pd.DataFrame,
    groups: pd.Series,
    dependency: pd.DataFrame,
    annotations: pd.DataFrame,
    disease_models: list[str],
    control_models: list[str],
    cfg: PipelineConfig,
    run: Run,
    tissues: pd.Series | None = None,
    case_label: str = "case",
    control_label: str = "control",
    n_shortlist: int = 20,
) -> dict[str, pd.DataFrame]:
    LOG.info("=== Stage 1: target identification ===")

    de = differential_expression(
        expression, groups, case_label=case_label, control_label=control_label
    )
    run.save_table(de, "target_differential_expression", "target_id")

    dep = dependency_summary(dependency, disease_models, control_models)
    run.save_table(dep, "target_dependency", "target_id")

    ann = assess_druggability(annotations)

    spec = None
    if tissues is not None:
        spec = expression_specificity(expression, tissues, target_tissue="disease_tissue")

    table = build_target_table(de, dep, ann, spec)
    scored = score_targets(table, cfg.target_id)
    run.save_table(scored, "target_scored", "target_id")

    short = shortlist(scored, n=n_shortlist)
    run.save_table(short, "target_shortlist", "target_id")

    n_sig = int(((de["fdr"] <= cfg.target_id.max_fdr)
                 & (de["log2fc"].abs() >= cfg.target_id.min_abs_log2fc)).sum())
    run.log_metrics("target_id", {
        "n_genes_tested": int(len(de)),
        "n_significant_de": n_sig,
        "n_selective_dependencies": int((dep["dep_mean_disease"]
                                         <= cfg.target_id.dependency_threshold).sum()),
        "n_common_essential": int(dep["is_common_essential"].sum()),
        "n_shortlisted": int(len(short)),
        "top_target": str(short.iloc[0]["gene"]) if len(short) else None,
        "top_target_score": float(short.iloc[0]["target_score"]) if len(short) else None,
    })

    LOG.info("top 5 targets:\n%s", short.head(5)[
        ["rank", "gene", "family", "target_score", "recommended_modality", "rationale"]
    ].to_string(index=False))

    return {"differential_expression": de, "dependency": dep,
            "scored": scored, "shortlist": short}
