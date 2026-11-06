"""End-to-end orchestration.

`run_demo` wires the three stages together on simulated data so the whole
thing is executable and inspectable. `run_from_files` is the same orchestration
against your own CSV/SDF inputs -- the only difference is where the data comes
from.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chem.standardize import standardize_table
from .config import PipelineConfig
from .data import simulate
from .data.loaders import load_assay_csv, load_library_smi
from .registry import Run, read_run_table, set_global_seed, setup_logging
from .stages.hit_discovery import train_activity_model, virtual_screen
from .stages.lead_opt import optimize_leads, train_admet_models
from .stages.target_id import run_target_identification

LOG = logging.getLogger("dd_pipeline.pipeline")

ADMET_ENDPOINTS = ["logS", "logD", "hlm_clint", "herg_pic50", "caco2_papp"]


def _prepare_assay(obs: pd.DataFrame, endpoints: list[str]) -> pd.DataFrame:
    """Standardise once, then aggregate every endpoint over replicates.

    Done per endpoint and re-joined, because a compound may be measured in the
    potency assay but not the solubility assay -- which is the normal state of
    a real program, not an error.
    """
    base, stats = standardize_table(obs, smiles_col="smiles", value_col=endpoints[0])
    LOG.info("standardisation stats: %s", stats.as_dict())
    key_cols = ["canonical_smiles", "inchikey"]

    merged = base[[c for c in key_cols if c in base.columns]
                  + ["compound_id", endpoints[0], f"{endpoints[0]}_spread", "n_replicates"]]
    for ep in endpoints[1:]:
        if ep not in obs.columns:
            continue
        sub, _ = standardize_table(obs, smiles_col="smiles", value_col=ep)
        cols = ["canonical_smiles", ep, f"{ep}_spread"]
        merged = merged.merge(sub[cols], on="canonical_smiles", how="outer")
    return merged.reset_index(drop=True)


def run_demo(cfg: PipelineConfig | None = None, run_id: str | None = None,
             library_size: int = 6000, n_measured: int = 1800) -> dict[str, Any]:
    """Full pipeline on simulated data.

    Returns a dict of result tables. All numbers produced here describe the
    pipeline, not chemistry -- the data is generated (see data/simulate.py).
    """
    cfg = cfg or PipelineConfig()
    set_global_seed(cfg.seed)
    run = Run.create(cfg.run_root, cfg.project, run_id)
    setup_logging(run)
    LOG.info("run directory: %s", run.root)
    cfg.to_yaml(run.root / "config.yaml")

    # ---------------- Stage 1: target identification ---------------------
    omics = simulate.simulate_omics(seed=cfg.seed)
    t_res = run_target_identification(
        expression=omics["expression"],
        groups=omics["groups"]["group"],
        dependency=omics["dependency"],
        annotations=omics["annotations"],
        disease_models=omics["disease_models"]["model"].tolist(),
        control_models=omics["control_models"]["model"].tolist(),
        cfg=cfg, run=run,
        tissues=omics["tissues"]["tissue"],
    )
    # Because the data is simulated we can check the ranking against the key.
    key = omics["answer_key"]
    t_res["recovery"] = _target_recovery(t_res["scored"], key, run,
                                         shortlist=t_res["shortlist"])

    # ---------------- Stage 2: hit discovery -----------------------------
    LOG.info("=== Stage 2: hit discovery ===")
    library = simulate.build_virtual_library(n=library_size, seed=cfg.seed)
    obs = simulate.simulate_assay_data(library, n_measured=n_measured, seed=cfg.seed)

    assay = _prepare_assay(obs, ["pIC50"] + ADMET_ENDPOINTS)
    run.save_table(assay, "assay_curated", "hit_discovery")

    model, featurizer, ad, act_metrics = train_activity_model(assay, cfg, run, "pIC50")

    lib_std, _ = standardize_table(library, smiles_col="smiles")
    known = set(assay["canonical_smiles"].dropna())
    hits = virtual_screen(lib_std, model, featurizer, ad, cfg, run, known_smiles=known)

    # ---------------- Stage 3: lead optimisation -------------------------
    predictor, admet_metrics = train_admet_models(
        assay, ADMET_ENDPOINTS, cfg, run, featurizer
    )
    predictor.add("pIC50", model, ad)
    lo_res = optimize_leads(hits, predictor, cfg, run)

    # ---------------- Retrospective check --------------------------------
    # Only possible with a simulated oracle. This is the number that tells you
    # whether the pipeline worked, and it is exactly the number you never get
    # to see in a real program until the compounds come back from assay.
    # The screened pool is the correct baseline for enrichment: it is what the
    # model actually ranked, after filters. Read it back from the run.
    screened = read_run_table(run.root, "virtual_screen_scored")
    if screened is not None and "is_known" in screened.columns:
        screened = screened[~screened["is_known"].astype(bool)]
    oracle = _oracle_check(hits, lo_res, cfg, run, screened=screened)

    manifest = run.finalize(cfg.model_dump(mode="json"))
    LOG.info("done. manifest: %s", manifest)

    return {
        "run": run,
        "targets": t_res,
        "assay": assay,
        "hits": hits,
        "lead_opt": lo_res,
        "oracle": oracle,
        "metrics": run.metrics,
    }


def _target_recovery(scored: pd.DataFrame, key: pd.DataFrame, run: Run,
                     shortlist: pd.DataFrame | None = None) -> dict[str, Any]:
    """Score the ranking against the planted answer key.

    Reported at two points, because they measure different things:

    * the **raw score ranking**, which is what the weighted composite alone
      achieves; and
    * the **shortlist**, after the hard gates in `targets.rank.shortlist`
      remove pan-essential and low-druggability genes.

    The difference between the two is the value the gates add. Undruggable
    transcription factors in particular score well on biology and are supposed
    to survive the ranking and then be removed by the druggability gate --
    reading only the raw ranking makes the gate look like it failed.
    """
    merged = scored.merge(key, on="gene", how="left")
    merged["category"] = merged["category"].fillna("background")
    true_ranks = merged.loc[merged["category"] == "true_target", "rank"]
    n_true = len(true_ranks)

    def _count(cat: str, within: int) -> int:
        return int((merged.loc[merged["category"] == cat, "rank"] <= within).sum())

    res: dict[str, Any] = {
        "n_true_targets": int(n_true),
        "n_true_in_top_20": int((true_ranks <= 20).sum()),
        "n_true_in_top_50": int((true_ranks <= 50).sum()),
        "median_true_rank": float(true_ranks.median()) if n_true else None,
        "mean_score_by_category": merged.groupby("category")["target_score"].mean().to_dict(),
        "n_pan_essential_decoys_in_top_20": _count("decoy_pan_essential", 20),
        "n_undruggable_decoys_in_top_20": _count("decoy_undruggable", 20),
    }

    if shortlist is not None and len(shortlist):
        sl = shortlist.merge(key, on="gene", how="left")
        sl["category"] = sl["category"].fillna("background")
        counts = sl["category"].value_counts().to_dict()
        res["shortlist_size"] = int(len(sl))
        res["shortlist_n_true_targets"] = int(counts.get("true_target", 0))
        res["shortlist_n_pan_essential_decoys"] = int(counts.get("decoy_pan_essential", 0))
        res["shortlist_n_undruggable_decoys"] = int(counts.get("decoy_undruggable", 0))
        res["shortlist_precision"] = float(
            res["shortlist_n_true_targets"] / max(len(sl), 1)
        )

    run.log_metrics("target_id_recovery", {k: v for k, v in res.items()
                                           if not isinstance(v, dict)})
    run.save_json(res, "target_recovery", "target_id")
    LOG.info("target recovery (raw ranking): %d/%d planted targets in the top 20 "
             "(median rank %s); decoys in top 20: %d pan-essential, %d undruggable",
             res["n_true_in_top_20"], n_true, res["median_true_rank"],
             res["n_pan_essential_decoys_in_top_20"], res["n_undruggable_decoys_in_top_20"])
    if "shortlist_size" in res:
        LOG.info("target recovery (after hard gates): %d/%d of the shortlist are planted "
                 "targets (precision %.2f); %d pan-essential and %d undruggable decoys "
                 "survived the gates",
                 res["shortlist_n_true_targets"], res["shortlist_size"],
                 res["shortlist_precision"], res["shortlist_n_pan_essential_decoys"],
                 res["shortlist_n_undruggable_decoys"])
    return res


def _oracle_check(hits: pd.DataFrame, lo_res: dict[str, pd.DataFrame],
                  cfg: PipelineConfig, run: Run,
                  screened: pd.DataFrame | None = None,
                  smiles_col: str = "canonical_smiles") -> dict[str, Any]:
    from .scoring.desirability import MPOScorer

    scorer = MPOScorer(cfg.lead_opt)
    res: dict[str, Any] = {}

    truth_hits = simulate.truth_for_smiles(hits["canonical_smiles"].tolist())
    if not truth_hits.empty:
        res["hit_list_true_pIC50_mean"] = float(truth_hits["true_pIC50"].mean())
        res["hit_list_true_pIC50_max"] = float(truth_hits["true_pIC50"].max())
        res["hit_list_true_hit_rate_at_6"] = float((truth_hits["true_pIC50"] >= 6.0).mean())
        res["hit_list_true_hit_rate_at_7"] = float((truth_hits["true_pIC50"] >= 7.0).mean())

    # Baseline: the hit rate from picking at random out of *the pool the model
    # actually ranked*.
    #
    # Getting this denominator wrong produces nonsense. An earlier version
    # generated a separate 2000-compound library as the baseline, which is not
    # the same distribution: build_virtual_library keeps the first N products
    # that pass the lead-like gate, so an n=2000 draw differs systematically
    # from an n=6000 one. That baseline reported ~14% of compounds at pIC50>=7
    # where the real screened pool had 1.3%, and the resulting "enrichment" of
    # 0.92 hid a genuine ~10x.
    if screened is not None and len(screened):
        pool_truth = simulate.truth_for_smiles(screened[smiles_col].tolist())
        if not pool_truth.empty:
            for thr in (6.0, 7.0):
                key = f"screened_pool_hit_rate_at_{int(thr)}"
                res[key] = float((pool_truth["true_pIC50"] >= thr).mean())
                if res[key] > 0:
                    res[f"screening_enrichment_at_{int(thr)}"] = (
                        res[f"hit_list_true_hit_rate_at_{int(thr)}"] / res[key]
                    )
            res["screened_pool_size"] = int(len(pool_truth))
            res["screened_pool_true_pIC50_mean"] = float(pool_truth["true_pIC50"].mean())

    cand = lo_res["candidates"]
    if len(cand):
        truth_c = simulate.truth_for_smiles(cand["smiles"].tolist())
        if not truth_c.empty:
            tc = truth_c.rename(columns=lambda c: c.replace("true_", ""))
            truth_c = truth_c.assign(true_mpo=scorer.score_frame(tc).score)
            # Join on SMILES rather than slicing: truth_for_smiles drops
            # anything unparsable, so positional alignment would quietly
            # compare different molecules and the diagnostic below -- whose
            # whole job is to detect over-optimisation -- would be meaningless.
            paired = cand[["smiles", "mpo_score"]].merge(
                truth_c[["smiles", "true_mpo", "true_pIC50"]], on="smiles", how="inner"
            )
            res["candidates_true_mpo_mean"] = float(paired["true_mpo"].mean())
            res["candidates_true_mpo_max"] = float(paired["true_mpo"].max())
            res["candidates_true_pIC50_mean"] = float(paired["true_pIC50"].mean())
            res["candidates_predicted_mpo_mean"] = float(paired["mpo_score"].mean())
            res["n_candidates_paired"] = int(len(paired))
            # Does the predicted MPO track the true MPO on the shortlist? A
            # near-zero or negative value is the signature of an optimiser
            # exploiting model error rather than finding better molecules.
            if len(paired) > 3 and paired["mpo_score"].nunique() > 1:
                from scipy import stats as _st
                res["candidate_mpo_pred_vs_true_spearman"] = float(
                    _st.spearmanr(paired["mpo_score"], paired["true_mpo"]).statistic
                )

    seeds = lo_res["seeds"]
    if len(seeds):
        ts = simulate.truth_for_smiles(seeds["smiles"].tolist())
        if not ts.empty:
            tsr = ts.rename(columns=lambda c: c.replace("true_", ""))
            res["seeds_true_mpo_mean"] = float(np.mean(scorer.score_frame(tsr).score))

    run.log_metrics("oracle_retrospective", res)
    run.save_json(res, "oracle_retrospective", "validation")
    LOG.info("ORACLE CHECK (simulated truth):")
    for k, v in res.items():
        LOG.info("  %-42s %s", k, f"{v:.4f}" if isinstance(v, float) else v)
    return res


def run_from_files(
    cfg: PipelineConfig,
    assay_csv: str | Path,
    library_path: str | Path,
    admet_csvs: dict[str, str | Path] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Stages 2-3 on your own data.

    assay_csv   -- potency measurements: smiles, value, units[, relation]
    library_path-- .smi/.csv of compounds to screen
    admet_csvs  -- {endpoint: path}, each with smiles + value on its own scale
    """
    set_global_seed(cfg.seed)
    run = Run.create(cfg.run_root, cfg.project, run_id)
    setup_logging(run)
    cfg.to_yaml(run.root / "config.yaml")

    raw = load_assay_csv(assay_csv)
    raw = raw.rename(columns={"pactivity": "pIC50"})
    frames = [raw[["smiles", "pIC50"]]]
    endpoints = ["pIC50"]
    for ep, path in (admet_csvs or {}).items():
        sub = pd.read_csv(path)
        if "smiles" not in sub.columns or ep not in sub.columns:
            raise ValueError(f"{path} must contain 'smiles' and '{ep}' columns")
        frames.append(sub[["smiles", ep]])
        endpoints.append(ep)

    obs = frames[0]
    for f in frames[1:]:
        obs = obs.merge(f, on="smiles", how="outer")
    obs["compound_id"] = [f"C{i:07d}" for i in range(len(obs))]

    assay = _prepare_assay(obs, endpoints)
    run.save_table(assay, "assay_curated", "hit_discovery")

    model, featurizer, ad, _ = train_activity_model(assay, cfg, run, "pIC50")

    library = load_library_smi(library_path)
    lib_std, _ = standardize_table(library, smiles_col="smiles")
    known = set(assay["canonical_smiles"].dropna())
    hits = virtual_screen(lib_std, model, featurizer, ad, cfg, run, known_smiles=known)

    predictor, _ = train_admet_models(
        assay, [e for e in endpoints if e != "pIC50"], cfg, run, featurizer
    )
    predictor.add("pIC50", model, ad)
    lo_res = optimize_leads(hits, predictor, cfg, run)

    run.finalize(cfg.model_dump(mode="json"))
    return {"run": run, "assay": assay, "hits": hits, "lead_opt": lo_res,
            "metrics": run.metrics}
