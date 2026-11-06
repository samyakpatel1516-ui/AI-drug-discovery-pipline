"""Run reporting: turn a manifest into something a project team will read.

Deliberately plain markdown. The point is that the numbers which qualify a
result travel with the result -- split strategy, baselines, applicability
domain coverage, calibration -- so nobody has to take the shortlist on trust.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOG = logging.getLogger("dd_pipeline.report")


from .registry import read_run_table as _read_table  # noqa: E402


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _metric_table(metrics: dict[str, Any], keys: list[tuple[str, str]]) -> list[str]:
    lines = ["| metric | value |", "| --- | --- |"]
    for key, label in keys:
        if key in metrics:
            lines.append(f"| {label} | {_fmt(metrics[key])} |")
    return lines


def build_report(run_root: str | Path) -> str:
    run_root = Path(run_root)
    manifest = json.loads((run_root / "manifest.json").read_text())
    m = manifest.get("metrics", {})
    cfg = manifest.get("config", {})

    out: list[str] = []
    out.append(f"# Discovery run report — {manifest.get('project')}")
    out.append("")
    out.append(f"Run directory: `{manifest.get('run_dir')}`  ")
    out.append(f"Duration: {manifest.get('duration_s')} s  ")
    out.append(f"Split strategy: **{cfg.get('split', {}).get('strategy')}** "
               f"(test {cfg.get('split', {}).get('test_frac')}, "
               f"valid {cfg.get('split', {}).get('valid_frac')})")
    out.append("")

    # ---- Stage 1 --------------------------------------------------------
    if "target_id" in m:
        t = m["target_id"]
        out.append("## Stage 1 — target identification")
        out.append("")
        out += _metric_table(t, [
            ("n_genes_tested", "genes tested"),
            ("n_significant_de", "significantly differentially expressed"),
            ("n_selective_dependencies", "selective dependencies"),
            ("n_common_essential", "pan-essential (excluded)"),
            ("n_shortlisted", "shortlisted"),
            ("top_target", "top target"),
            ("top_target_score", "top target score"),
        ])
        out.append("")
        short = _read_table(run_root, "target_shortlist")
        if short is not None and len(short):
            cols = [c for c in ("rank", "gene", "family", "target_score",
                                "recommended_modality", "rationale") if c in short.columns]
            out.append("Top 10 targets:")
            out.append("")
            out.append(short.head(10)[cols].to_markdown(index=False))
            out.append("")
        if "target_id_recovery" in m:
            r = m["target_id_recovery"]
            out.append("**Retrospective check against the simulated answer key** "
                       "(only possible because the data is simulated):")
            out.append("")
            out += _metric_table(r, [
                ("n_true_targets", "planted targets"),
                ("n_true_in_top_20", "recovered in top 20 of the raw ranking"),
                ("median_true_rank", "median rank of planted targets"),
                ("n_pan_essential_decoys_in_top_20", "pan-essential decoys in raw top 20"),
                ("n_undruggable_decoys_in_top_20", "undruggable decoys in raw top 20"),
                ("shortlist_size", "shortlist size (after hard gates)"),
                ("shortlist_n_true_targets", "planted targets on the shortlist"),
                ("shortlist_precision", "shortlist precision"),
                ("shortlist_n_pan_essential_decoys", "pan-essential decoys surviving gates"),
                ("shortlist_n_undruggable_decoys", "undruggable decoys surviving gates"),
            ])
            out.append("")
            if r.get("n_undruggable_decoys_in_top_20", 0) > 0:
                out.append("- Undruggable decoys score well on biology and are *meant* to "
                           "rank highly; the druggability gate is what removes them. Read "
                           "the shortlist rows, not the raw ranking, to judge that gate.")
                out.append("")

    # ---- Stage 2 --------------------------------------------------------
    act_key = next((k for k in m if k.startswith("activity_model_")), None)
    if act_key:
        a = m[act_key]
        out.append("## Stage 2 — activity model and virtual screen")
        out.append("")
        out.append("### Model quality on the held-out split")
        out.append("")
        out += _metric_table(a, [
            ("n", "test compounds"),
            ("train_scaffolds", "train scaffolds"),
            ("test_scaffolds", "test scaffolds"),
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("r2", "R²"),
            ("spearman", "Spearman ρ"),
            ("frac_within_0.5_log", "fraction within 0.5 log"),
            ("baseline_constant_rmse", "baseline: constant predictor RMSE"),
            ("baseline_1nn_rmse", "baseline: 1-NN RMSE"),
            ("rmse_over_assay_noise", "RMSE ÷ assay noise"),
            ("conformal_coverage_90", "conformal 90% coverage (target 0.90)"),
            ("mean_interval_width", "mean 90% interval width"),
            ("unc_error_spearman", "uncertainty vs error ρ"),
            ("active_prevalence", "active prevalence in test set"),
            ("ef_ceiling", "EF ceiling (1/prevalence)"),
            ("ef_1pct", "EF @ 1%"),
            ("pr_auc_at_threshold", "PR-AUC at the active threshold"),
        ])
        out.append("")
        notes = []
        if a.get("baseline_1nn_rmse") and a["rmse"] >= a["baseline_1nn_rmse"]:
            notes.append("- :warning: **The model does not beat 1-nearest-neighbour.** "
                         "It has learned nothing beyond chemical similarity.")
        if a.get("conformal_coverage_90") and abs(a["conformal_coverage_90"] - 0.9) > 0.07:
            notes.append(f"- :warning: Conformal coverage is "
                         f"{a['conformal_coverage_90']:.2f}, not ~0.90; the prediction "
                         "intervals should not be trusted.")
        if a.get("active_prevalence", 0) > 0.5:
            notes.append("- Enrichment factors are saturated (the test set is mostly "
                         "active); read Spearman and PR-AUC instead.")
        if a.get("unc_error_spearman") is not None and a["unc_error_spearman"] < 0.15:
            notes.append("- Predicted uncertainty correlates only weakly with actual "
                         "error, so it is a weak basis for active learning here. The "
                         "applicability domain remains the reliable guard.")
        if a.get("rmse_over_assay_noise") and a["rmse_over_assay_noise"] < 1.2:
            notes.append("- Model error is close to the assay noise floor; more data "
                         "of the same kind will give little further improvement.")
        if notes:
            out.append("Reading of the above:")
            out.append("")
            out += notes
            out.append("")

    if "hit_discovery" in m:
        h = m["hit_discovery"]
        out.append("### Virtual screen")
        out.append("")
        out += _metric_table(h, [
            ("n_library", "library size"),
            ("n_passed_filters", "passed alerts + property gates"),
            ("filter_pass_rate", "filter pass rate"),
            ("n_above_threshold", "above activity threshold"),
            ("n_hits_selected", "hits selected"),
            ("n_unique_scaffolds_in_hits", "unique scaffolds among hits"),
            ("frac_hits_in_domain", "fraction of hits in applicability domain"),
            ("mean_hit_pred_activity", "mean predicted activity of hits"),
            ("mean_hit_sa_score", "mean synthetic accessibility of hits"),
        ])
        out.append("")

    # ---- Stage 3 --------------------------------------------------------
    admet_keys = sorted(k for k in m if k.startswith("admet_"))
    if admet_keys:
        out.append("## Stage 3 — ADMET models and lead optimisation")
        out.append("")
        rows = []
        for k in admet_keys:
            d = m[k]
            rows.append({
                "endpoint": k.replace("admet_", ""),
                "n_test": d.get("n"),
                "RMSE": _fmt(d.get("rmse")),
                "Spearman": _fmt(d.get("spearman")),
                "1-NN RMSE": _fmt(d.get("baseline_1nn_rmse")),
                "beats 1-NN": ("yes" if d.get("baseline_1nn_rmse", 0) > d.get("rmse", 1e9)
                               else "NO"),
                "90% coverage": _fmt(d.get("conformal_coverage_90")),
            })
        out.append(pd.DataFrame(rows).to_markdown(index=False))
        out.append("")

    if "lead_opt" in m:
        lo = m["lead_opt"]
        out.append("### Optimisation outcome")
        out.append("")
        out += _metric_table(lo, [
            ("n_seeds", "starting hits"),
            ("n_analogs_generated", "admissible analogs generated"),
            ("n_scored_total", "molecules scored"),
            ("n_pareto_front", "on the Pareto front"),
            ("n_final_candidates", "final candidates"),
            ("n_unique_scaffolds_final", "unique scaffolds in the shortlist"),
            ("best_seed_mpo", "best MPO among starting hits"),
            ("best_analog_mpo", "best MPO among generated analogs"),
            ("best_candidate_mpo", "best MPO in the shortlist"),
            ("mpo_improvement_best", "MPO improvement (best)"),
            ("mean_seed_mpo", "mean MPO of starting hits"),
            ("mean_candidate_mpo", "mean MPO of the shortlist"),
            ("mpo_improvement_mean", "MPO improvement (mean)"),
            ("n_analogs_beating_best_seed", "analogs beating the best starting hit"),
            ("uncertainty_penalty_beta", "uncertainty penalty β (σ)"),
            ("mean_optimism_gap", "mean optimism gap (optimistic − conservative MPO)"),
            ("min_objective_domain_coverage", "worst objective's domain coverage"),
        ])
        worst = lo.get("min_objective_domain_coverage")
        if worst is not None and worst < 0.5:
            bad = [k.replace("domain_coverage_", "") for k, v in lo.items()
                   if k.startswith("domain_coverage_") and isinstance(v, (int, float))
                   and v < 0.5]
            out.append(f"- :warning: Objective(s) {', '.join(bad)} are in-domain for under "
                       "half the molecules scored. Those desirability terms are "
                       "extrapolation, so the combined MPO score is only partly "
                       "meaningful — it is not a defect of the optimiser but of the "
                       "endpoint data's coverage of this chemistry.")
            out.append("")
        out.append("")
        if lo.get("n_analogs_beating_best_seed") == 0:
            out.append("- The best molecule is still a starting hit: generation broadened "
                       "the viable set rather than finding a new peak. Compare the mean "
                       "improvement, which is what a 25-compound shortlist depends on.")
            out.append("")
        cand = _read_table(run_root, "final_candidates")
        if cand is not None and len(cand):
            cols = [c for c in ("candidate_rank", "smiles", "mpo_score", "pIC50", "logS",
                                "logD", "hlm_clint", "herg_pic50", "caco2_papp",
                                "sa_score", "ligand_efficiency", "on_pareto_front")
                    if c in cand.columns]
            out.append("Top 10 candidates:")
            out.append("")
            disp = cand.head(10)[cols].copy()
            for c in disp.columns:
                if disp[c].dtype.kind == "f":
                    disp[c] = disp[c].round(2)
            out.append(disp.to_markdown(index=False))
            out.append("")

    # ---- Oracle ---------------------------------------------------------
    if "oracle_retrospective" in m:
        o = m["oracle_retrospective"]
        out.append("## Retrospective validation against simulated truth")
        out.append("")
        out.append("These numbers exist only because the demo data is generated. In a "
                   "real programme you learn them when the compounds come back from "
                   "assay, which is the whole reason the sections above matter.")
        out.append("")
        out += _metric_table(o, [
            ("screened_pool_size", "compounds the model ranked (baseline denominator)"),
            ("hit_list_true_hit_rate_at_6", "true hit rate of the hit list (pIC50 ≥ 6)"),
            ("screened_pool_hit_rate_at_6", "true hit rate of a random pool pick (≥ 6)"),
            ("screening_enrichment_at_6", "screening enrichment at pIC50 ≥ 6"),
            ("hit_list_true_hit_rate_at_7", "true hit rate of the hit list (pIC50 ≥ 7)"),
            ("screened_pool_hit_rate_at_7", "true hit rate of a random pool pick (≥ 7)"),
            ("screening_enrichment_at_7", "screening enrichment at pIC50 ≥ 7"),
            ("seeds_true_mpo_mean", "mean true MPO of the starting hits"),
            ("candidates_true_mpo_mean", "mean true MPO of the final candidates"),
            ("candidates_true_mpo_max", "best true MPO of the final candidates"),
            ("candidate_mpo_pred_vs_true_spearman", "predicted vs true MPO ρ on shortlist"),
        ])
        out.append("")
        rho = o.get("candidate_mpo_pred_vs_true_spearman")
        if rho is not None and rho < 0.4:
            out.append(f"- :warning: Predicted MPO correlates only {rho:+.2f} with true MPO "
                       "on the shortlist. That is the signature of an optimiser exploiting "
                       "model error: the search may still have improved the molecules on "
                       "average while the *ordering* within the shortlist is noise. Raise "
                       "`lead_opt.uncertainty_penalty`, tighten the applicability domain, "
                       "or reduce the number of generations.")
            out.append("")

    # ---- Provenance -----------------------------------------------------
    out.append("## Provenance")
    out.append("")
    out.append(f"{len(manifest.get('artifacts', []))} artifacts recorded with SHA-256 "
               "hashes in `manifest.json`; environment and package versions in "
               "`environment.json`; exact configuration in `config.yaml`.")
    out.append("")
    return "\n".join(out)


def write_report(run_root: str | Path, filename: str = "report.md") -> Path:
    run_root = Path(run_root)
    text = build_report(run_root)
    out = run_root / "reports" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    LOG.info("report -> %s", out)
    return out
