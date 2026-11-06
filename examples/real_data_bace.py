"""Validate the pipeline on real, published, measured data.

No simulation anywhere in this script. Three public benchmark datasets:

    BACE-1 pIC50   1500 compounds   target potency (Alzheimer's beta-secretase)
    logD7.4        4175 compounds   lipophilicity, ChEMBL-curated
    logS           1011 compounds   aqueous solubility (Delaney/ESOL)

Four experiments, in order of how much they tell you:

  1. **Split protocol.** Train each endpoint under a random split and a
     scaffold split. The gap between them is the single most useful number in
     this script: it is how much of a model's apparent quality comes from
     testing it on analogs of its training set.

  2. **Retrospective virtual screen, measured only.** Hold out a
     scaffold-disjoint slice of BACE-1, treat its measured potencies as
     unknown, screen it, then score the selection against the real numbers.
     Honest but pessimistic about enrichment: everything in this set came from
     a BACE paper, so actives are ~40% of it and enrichment factor is pinned
     near its ceiling.

  3. **Retrospective screen with decoys.** Same holdout, diluted with
     property-matched compounds from the lipophilicity set that nobody has
     tested against BACE-1. Restores a realistic active prevalence so
     enrichment means something. Costs an assumption: decoys are *presumed*
     inactive, not measured inactive.

  4. **Multi-parameter optimisation** of real hits against three models
     trained on three disjoint real datasets.

Run:  python examples/real_data_bace.py
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from dd_pipeline.chem.featurize import Featurizer
from dd_pipeline.chem.splits import make_split, murcko_scaffold
from dd_pipeline.config import (
    LeadOptConfig,
    ObjectiveSpec,
    PipelineConfig,
    SplitConfig,
)
from dd_pipeline.data.public import DATASETS, build_decoy_library, load_dataset
from dd_pipeline.models.applicability import ApplicabilityDomain
from dd_pipeline.models.evaluate import baseline_metrics, enrichment_factor, evaluate_regression
from dd_pipeline.models.qsar import EnsembleQSAR
from dd_pipeline.registry import Run, set_global_seed, setup_logging
from dd_pipeline.stages.hit_discovery import virtual_screen
from dd_pipeline.stages.lead_opt import PropertyPredictor, optimize_leads

LOG = logging.getLogger("real_data")

# BACE-1: 100 nM (pIC50 7.0) is a reasonable "worth progressing" line for a
# lead, and 10 nM (8.0) a stretch goal. Both are reported.
ACTIVE_THRESHOLDS = (7.0, 8.0)


def build_config() -> PipelineConfig:
    cfg = PipelineConfig(project="bace1-real-data", seed=20261106)
    cfg.activity_model.n_estimators = 800
    cfg.activity_model.n_ensemble = 5
    cfg.admet_model.n_estimators = 800
    cfg.admet_model.n_ensemble = 5

    # Potency gate for hit selection, in real pIC50 units.
    cfg.hit_discovery.min_predicted_activity = 7.0
    cfg.hit_discovery.n_hits = 100
    cfg.hit_discovery.exploration_frac = 0.2
    # BACE-1 inhibitors are famously large and peptidomimetic; the default
    # lead-like window would reject most of the real chemical matter. Widening
    # it is the honest move, and a good illustration that property gates are
    # programme-specific rather than universal.
    cfg.hit_discovery.mw_range = (250.0, 750.0)
    cfg.hit_discovery.clogp_range = (-1.0, 6.5)
    cfg.hit_discovery.max_rotatable_bonds = 15
    cfg.hit_discovery.max_hbd = 7
    cfg.hit_discovery.max_tpsa = 200.0
    cfg.hit_discovery.apply_brenk = False      # too aggressive for this series
    cfg.hit_discovery.max_pairwise_similarity = 0.75

    # Only the three endpoints we actually have real data for.
    cfg.lead_opt = LeadOptConfig(
        objectives=[
            ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=7.0, high=9.5,
                          weight=3.0, critical=True),
            ObjectiveSpec(endpoint="logS", kind="higher_better", low=-5.5, high=-3.5,
                          weight=1.5),
            ObjectiveSpec(endpoint="logD", kind="window", low=1.0, high=3.5, weight=1.5),
        ],
        n_seeds=15,
        max_generations=3,
        n_final_candidates=20,
        max_sa_score=6.0,          # real BACE chemistry is not simple
        min_similarity_to_seed=0.35,
        uncertainty_penalty=1.0,
    )
    return cfg


# ---------------------------------------------------------------------------
# Experiment 1: how much does the split protocol matter?
# ---------------------------------------------------------------------------
def experiment_split_protocol(cfg: PipelineConfig, run: Run) -> pd.DataFrame:
    LOG.info("=" * 78)
    LOG.info("EXPERIMENT 1: random vs scaffold split on three real endpoints")
    LOG.info("=" * 78)
    rows = []
    featurizer = Featurizer(cfg.featurizer)

    for ds_name, spec in DATASETS.items():
        df, prov = load_dataset(ds_name)
        run.save_json(prov, f"provenance_{ds_name}", "data")
        endpoint = spec.endpoint
        fm = featurizer.transform(df["canonical_smiles"].tolist())
        y = df[endpoint].to_numpy(dtype=float)

        for strategy in ("random", "scaffold"):
            split_cfg = SplitConfig(strategy=strategy, test_frac=0.2, valid_frac=0.1,
                                    seed=cfg.seed)
            tr, va, te = make_split(df, split_cfg, smiles_col="canonical_smiles")
            model = EnsembleQSAR(cfg.activity_model, endpoint=endpoint)
            model.fit(fm.X[tr], y[tr], fm.X[va], y[va], feature_names=fm.feature_names)

            pred = model.predict(fm.X[te])
            thr = 7.0 if endpoint == "pIC50" else None
            m = evaluate_regression(y[te], pred, active_threshold=thr)
            fps = featurizer.bit_fingerprints(df["canonical_smiles"].tolist())
            m.update(baseline_metrics(y[tr], y[te],
                                      [fps[i] for i in tr], [fps[i] for i in te]))
            rows.append({
                "dataset": ds_name,
                "endpoint": endpoint,
                "n_total": len(df),
                "split": strategy,
                "n_test": int(len(te)),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "spearman": m["spearman"],
                "baseline_constant_rmse": m.get("baseline_constant_rmse"),
                "baseline_1nn_rmse": m.get("baseline_1nn_rmse"),
                "conformal_coverage_90": m.get("conformal_coverage_90"),
                "frac_within_0.5": m.get("frac_within_0.5_log"),
            })
            LOG.info(
                "%-14s %-8s split: RMSE=%.3f  R2=%+.3f  rho=%.3f  "
                "(constant %.3f, 1-NN %.3f)  coverage=%.3f",
                ds_name, strategy, m["rmse"], m["r2"], m["spearman"],
                m.get("baseline_constant_rmse", float("nan")),
                m.get("baseline_1nn_rmse", float("nan")),
                m.get("conformal_coverage_90", float("nan")),
            )

    out = pd.DataFrame(rows)
    out["rmse_penalty_vs_random"] = out.groupby("dataset")["rmse"].transform(
        lambda s: s - s.iloc[0]
    )
    out["beats_1nn"] = out["rmse"] < out["baseline_1nn_rmse"]
    run.save_table(out, "experiment1_split_protocol", "benchmark")
    return out


# ---------------------------------------------------------------------------
# Experiments 2 & 3: retrospective virtual screening on real BACE-1 data
# ---------------------------------------------------------------------------
def experiment_retrospective_screen(
    cfg: PipelineConfig, run: Run
) -> tuple[dict, pd.DataFrame, EnsembleQSAR, Featurizer, ApplicabilityDomain, pd.DataFrame]:
    LOG.info("=" * 78)
    LOG.info("EXPERIMENTS 2-3: retrospective virtual screen, real measured BACE-1")
    LOG.info("=" * 78)

    bace, _ = load_dataset("bace")
    featurizer = Featurizer(cfg.featurizer)

    # Scaffold-disjoint holdout plays the part of an unscreened library.
    split_cfg = SplitConfig(strategy="scaffold", test_frac=0.25, valid_frac=0.10,
                            seed=cfg.seed)
    tr, va, te = make_split(bace, split_cfg, smiles_col="canonical_smiles")
    train = bace.iloc[np.concatenate([tr, va])].reset_index(drop=True)
    library = bace.iloc[te].reset_index(drop=True)
    LOG.info("train+valid=%d, screening holdout=%d (scaffold-disjoint)",
             len(train), len(library))

    fm_tr = featurizer.transform(train["canonical_smiles"].tolist())
    y_tr = train["pIC50"].to_numpy(dtype=float)
    n_va = len(va)
    model = EnsembleQSAR(cfg.activity_model, endpoint="pIC50")
    model.fit(fm_tr.X[:-n_va], y_tr[:-n_va], fm_tr.X[-n_va:], y_tr[-n_va:],
              feature_names=fm_tr.feature_names)

    ad = ApplicabilityDomain(cfg.applicability, featurizer)
    n_fp = cfg.featurizer.fp_bits
    ad.fit(train["canonical_smiles"].tolist(), fm_tr.X[:, n_fp:])
    model.card.split_strategy = "scaffold (holdout used as screening library)"
    model.card.notes = (
        "Trained on real measured BACE-1 pIC50 from MoleculeNet. The held-out "
        "scaffolds were screened prospectively-by-simulation: predictions were "
        "made before the measured values were consulted."
    )
    model.save(run.path("models", "bace1_pIC50.pkl"))

    res: dict = {}

    # ---- Experiment 2: measured-only -----------------------------------
    fm_lib = featurizer.transform(library["canonical_smiles"].tolist())
    pred = model.predict(fm_lib.X)
    lib = library.copy()
    lib["pred_activity"] = pred.mean
    lib["pred_std"] = pred.std
    dom = ad.assess(lib["canonical_smiles"].tolist(), fm_lib.X[:, n_fp:])
    for k, v in dom.items():
        lib[k] = v

    m = evaluate_regression(lib["pIC50"].to_numpy(dtype=float), pred, active_threshold=7.0)
    res["measured_only"] = {
        "n_library": int(len(lib)),
        "rmse": m["rmse"], "spearman": m["spearman"], "r2": m["r2"],
        "active_prevalence": m.get("active_prevalence"),
        "ef_ceiling": m.get("ef_ceiling"),
        "frac_in_domain": float(lib["in_domain"].mean()),
    }
    for thr in ACTIVE_THRESHOLDS:
        for frac in (0.05, 0.10, 0.20):
            res["measured_only"][f"ef_{int(frac*100)}pct_at_pIC50_{thr:g}"] = enrichment_factor(
                lib["pIC50"].to_numpy(dtype=float), lib["pred_activity"].to_numpy(),
                frac=frac, threshold=thr, warn=False,
            )
    LOG.info("Exp 2 (measured only): n=%d  RMSE=%.3f  rho=%.3f  prevalence(>=7)=%.2f "
             "ceiling=%.2f  in-domain=%.2f",
             len(lib), m["rmse"], m["spearman"], m.get("active_prevalence", float("nan")),
             m.get("ef_ceiling", float("nan")), res["measured_only"]["frac_in_domain"])
    run.save_table(lib, "experiment2_measured_screen", "benchmark")

    # ---- Experiment 3: decoy-augmented ---------------------------------
    lipo, _ = load_dataset("lipophilicity")
    decoy_lib = build_decoy_library(
        actives=library[["canonical_smiles"]],
        decoy_pool=lipo[["canonical_smiles"]],
        n_decoys_per_active=10,
        seed=cfg.seed,
    )
    truth = dict(zip(library["canonical_smiles"], library["pIC50"]))

    screened = virtual_screen(decoy_lib, model, featurizer, ad, cfg, run,
                              known_smiles=set(train["canonical_smiles"]))
    scored = pd.read_parquet(run.root / "data" / "virtual_screen_scored.parquet")

    # Decoys are treated as inactive; measured compounds use their real value.
    scored["measured_pIC50"] = scored["canonical_smiles"].map(truth)
    scored["is_measured_active"] = scored["measured_pIC50"].notna()

    exp3: dict = {"n_library": int(len(decoy_lib)), "n_scored": int(len(scored))}
    for thr in ACTIVE_THRESHOLDS:
        labels = (scored["measured_pIC50"].fillna(-99) >= thr).to_numpy()
        prevalence = float(labels.mean())
        exp3[f"prevalence_at_{thr:g}"] = prevalence
        exp3[f"ef_ceiling_at_{thr:g}"] = float(1 / prevalence) if prevalence else float("nan")
        for frac in (0.01, 0.05, 0.10):
            k = max(1, int(np.ceil(frac * len(scored))))
            top = scored.nlargest(k, "pred_activity")
            rate = float((top["measured_pIC50"].fillna(-99) >= thr).mean())
            exp3[f"ef_{int(frac*100)}pct_at_{thr:g}"] = (
                rate / prevalence if prevalence else float("nan")
            )
        # What the pipeline's actual selection (filters + AD + diversity) delivers.
        sel = screened.merge(
            scored[["canonical_smiles", "measured_pIC50"]], on="canonical_smiles", how="left"
        )
        sel_rate = float((sel["measured_pIC50"].fillna(-99) >= thr).mean())
        exp3[f"delivered_hitlist_rate_at_{thr:g}"] = sel_rate
        exp3[f"delivered_hitlist_ef_at_{thr:g}"] = (
            sel_rate / prevalence if prevalence else float("nan")
        )
    res["with_decoys"] = exp3
    LOG.info("Exp 3 (with decoys): library=%d  prevalence(>=7)=%.4f  "
             "EF@1%%=%.1f  EF@5%%=%.1f  delivered hit list EF=%.1f",
             exp3["n_library"], exp3["prevalence_at_7"],
             exp3["ef_1pct_at_7"], exp3["ef_5pct_at_7"], exp3["delivered_hitlist_ef_at_7"])

    run.log_metrics("retrospective_screen", {
        f"{k}_{kk}": vv for k, v in res.items() for kk, vv in v.items()
        if isinstance(vv, (int, float))
    })
    run.save_json(res, "retrospective_screen", "benchmark")
    return res, screened, model, featurizer, ad, train


# ---------------------------------------------------------------------------
# Experiment 4: MPO over three real endpoints
# ---------------------------------------------------------------------------
def experiment_lead_optimization(
    cfg: PipelineConfig, run: Run, hits: pd.DataFrame,
    activity_model: EnsembleQSAR, featurizer: Featurizer, ad: ApplicabilityDomain,
) -> dict:
    LOG.info("=" * 78)
    LOG.info("EXPERIMENT 4: MPO with three models trained on three real datasets")
    LOG.info("=" * 78)

    predictor = PropertyPredictor(featurizer, cfg)
    predictor.add("pIC50", activity_model, ad)

    for ds_name in ("lipophilicity", "esol"):
        df, _ = load_dataset(ds_name)
        endpoint = DATASETS[ds_name].endpoint
        split_cfg = SplitConfig(strategy="scaffold", test_frac=0.15, valid_frac=0.10,
                                seed=cfg.seed)
        tr, va, te = make_split(df, split_cfg, smiles_col="canonical_smiles")
        fm = featurizer.transform(df["canonical_smiles"].tolist())
        y = df[endpoint].to_numpy(dtype=float)

        m = EnsembleQSAR(cfg.admet_model, endpoint=endpoint)
        m.fit(fm.X[tr], y[tr], fm.X[va], y[va], feature_names=fm.feature_names)
        dom = ApplicabilityDomain(cfg.applicability, featurizer)
        dom.fit(df.loc[tr, "canonical_smiles"].tolist(), fm.X[tr][:, cfg.featurizer.fp_bits:])

        test_metrics = evaluate_regression(y[te], m.predict(fm.X[te]))
        m.card.metrics = {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}
        m.card.split_strategy = "scaffold"
        m.save(run.path("models", f"{endpoint}.pkl"))
        predictor.add(endpoint, m, dom)
        run.log_metrics(f"admet_{endpoint}_real", test_metrics)
        LOG.info("%-6s model on real data: n=%d  RMSE=%.3f  rho=%.3f",
                 endpoint, len(df), test_metrics["rmse"], test_metrics["spearman"])

    results = optimize_leads(hits, predictor, cfg, run)
    cand = results["candidates"]

    # How far outside the ADMET models' training space are these BACE hits?
    # This is the number that decides whether the MPO is meaningful at all.
    frac_dom = {}
    if len(cand):
        for ep in ("logD", "logS", "pIC50"):
            col = f"{ep}_in_domain"
            if col in cand.columns:
                frac_dom[ep] = float(cand[col].fillna(False).astype(bool).mean())
        LOG.info("fraction of final candidates inside each model's domain: %s",
                 {k: round(v, 3) for k, v in frac_dom.items()})
    run.save_json(frac_dom, "candidate_domain_coverage", "lead_opt")
    return {"n_candidates": int(len(cand)), "domain_coverage": frac_dom}


def main() -> None:
    setup_logging(level=logging.INFO)
    for noisy in ("dd_pipeline.chem.featurize", "dd_pipeline.models.applicability"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = build_config()
    set_global_seed(cfg.seed)
    run = Run.create(cfg.run_root, cfg.project)
    setup_logging(run)
    cfg.to_yaml(run.root / "config.yaml")
    LOG.info("run directory: %s", run.root)

    split_table = experiment_split_protocol(cfg, run)
    screen_res, hits, model, featurizer, ad, _train = experiment_retrospective_screen(cfg, run)
    lo_res = experiment_lead_optimization(cfg, run, hits, model, featurizer, ad)

    run.finalize(cfg.model_dump(mode="json"))

    print("\n" + "=" * 78)
    print("REAL-DATA RESULTS (no simulation anywhere)")
    print("=" * 78)
    cols = ["dataset", "endpoint", "n_total", "split", "rmse", "r2", "spearman",
            "baseline_1nn_rmse", "beats_1nn", "conformal_coverage_90"]
    print("\nExperiment 1 -- split protocol:\n")
    print(split_table[cols].round(3).to_string(index=False))
    print("\nExperiment 2/3 -- retrospective virtual screen on BACE-1:\n")
    print(json.dumps(screen_res, indent=2, default=lambda o: round(float(o), 4)))
    print("\nExperiment 4 -- lead optimisation:\n")
    print(json.dumps(lo_res, indent=2, default=str))
    print(f"\nrun directory: {run.root}")


if __name__ == "__main__":
    main()
