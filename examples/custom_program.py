"""Template: wire the pipeline to your own program.

Copy this, point the three loaders at your data, edit the objectives, run it.
The only genuinely program-specific decisions are marked (1)-(4).
"""
import logging

from dd_pipeline.chem.standardize import standardize_table
from dd_pipeline.config import (
    LeadOptConfig,
    ObjectiveSpec,
    PipelineConfig,
    SplitConfig,
)
from dd_pipeline.data.loaders import load_assay_csv, load_library_smi
from dd_pipeline.registry import Run, set_global_seed, setup_logging
from dd_pipeline.report import write_report
from dd_pipeline.stages.hit_discovery import train_activity_model, virtual_screen
from dd_pipeline.stages.lead_opt import optimize_leads, train_admet_models


def build_config() -> PipelineConfig:
    cfg = PipelineConfig(project="my-program", seed=20260101)

    # (1) Validation strategy. If your assay table has dates, use a temporal
    # split -- it is the closest thing to prospective performance you can
    # measure, and the number you should quote to a project team.
    cfg.split = SplitConfig(strategy="scaffold", test_frac=0.2, valid_frac=0.1)
    # cfg.split = SplitConfig(strategy="temporal", date_column="assay_date")

    # (2) The target product profile, as desirability objectives. These bounds
    # are the single most important thing in this file: they encode what the
    # programme is actually trying to achieve, and the optimiser will exploit
    # anything you leave out. Mark as `critical` only what genuinely kills a
    # compound.
    cfg.lead_opt = LeadOptConfig(
        objectives=[
            ObjectiveSpec(endpoint="pIC50", kind="higher_better",
                          low=7.0, high=9.5, weight=3.0, critical=True),
            ObjectiveSpec(endpoint="logS", kind="higher_better",
                          low=-5.5, high=-3.5, weight=1.5),
            ObjectiveSpec(endpoint="logD", kind="window",
                          low=1.5, high=3.0, weight=1.0),
            ObjectiveSpec(endpoint="hlm_clint", kind="lower_better",
                          low=8.0, high=40.0, weight=2.0),
            ObjectiveSpec(endpoint="herg_pic50", kind="lower_better",
                          low=4.5, high=5.5, weight=2.5, critical=True),
            # Add your own: ppb, CYP3A4 inhibition, MDCK efflux ratio, ...
        ],
        n_seeds=25,
        max_generations=4,
        n_final_candidates=40,
        max_sa_score=4.0,
    )

    # (3) Screening economics: how many compounds you can actually buy or make,
    # and how much of that budget goes to learning rather than to winning.
    cfg.hit_discovery.n_hits = 384            # one plate
    cfg.hit_discovery.exploration_frac = 0.25
    cfg.hit_discovery.min_predicted_activity = 6.5
    return cfg


def main() -> None:
    setup_logging(level=logging.INFO)
    cfg = build_config()
    set_global_seed(cfg.seed)
    run = Run.create(cfg.run_root, cfg.project)
    cfg.to_yaml(run.root / "config.yaml")

    # (4) Your data. load_assay_csv normalises units to p-scale and drops
    # qualified ('>') values rather than treating them as exact.
    potency = load_assay_csv("data/potency.csv", value_col="IC50_nM", units_col="units")
    potency = potency.rename(columns={"pactivity": "pIC50"})

    assay, stats = standardize_table(potency, smiles_col="smiles", value_col="pIC50")
    logging.info("curation: %s", stats.as_dict())
    run.save_table(assay, "assay_curated", "hit_discovery")

    model, featurizer, ad, _ = train_activity_model(assay, cfg, run, "pIC50")

    library = load_library_smi("data/vendor_library.smi")
    library, _ = standardize_table(library, smiles_col="smiles")
    hits = virtual_screen(
        library, model, featurizer, ad, cfg, run,
        known_smiles=set(assay["canonical_smiles"].dropna()),
    )

    admet_endpoints = [o.endpoint for o in cfg.lead_opt.objectives if o.endpoint != "pIC50"]
    predictor, _ = train_admet_models(assay, admet_endpoints, cfg, run, featurizer)
    predictor.add("pIC50", model, ad)

    results = optimize_leads(hits, predictor, cfg, run)
    run.finalize(cfg.model_dump(mode="json"))
    write_report(run.root)

    print(f"{len(results['candidates'])} candidates -> {run.root}")
    print("Check before acting on the shortlist:")
    print("  - does the model beat baseline_1nn_rmse in the report?")
    print("  - is conformal_coverage_90 near 0.90?")
    print("  - are the candidates in_domain?")


if __name__ == "__main__":
    main()
