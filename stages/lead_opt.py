"""Stage 3: ADMET modelling and multi-objective lead optimisation.

The structure here reflects the actual job. You have a handful of hits. You
need candidates that are potent *and* soluble *and* permeable *and* not
cleared in minutes *and* not hERG-active. So:

  1. train a model per ADMET endpoint (same machinery as activity)
  2. predict every endpoint for any molecule, and compute an MPO score
  3. evolve analogs of the hits against that MPO score
  4. report the Pareto front, not just the top of a ranked list
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

from ..chem.enumerate import GAConfig, GeneticOptimizer
from ..chem.featurize import Featurizer
from ..chem.filters import StructuralAlertFilter, property_gate
from ..chem.splits import make_split, murcko_scaffold
from ..chem.synth import sa_score_smiles
from ..config import PipelineConfig
from ..models.applicability import ApplicabilityDomain
from ..models.evaluate import baseline_metrics, evaluate_regression
from ..models.qsar import EnsembleQSAR
from ..registry import Run
from ..scoring.desirability import (
    MPOScorer,
    ligand_efficiency,
    lipophilic_efficiency,
    pessimistic_frame,
)
from ..scoring.pareto import crowding_distance, pareto_front, pareto_ranks

LOG = logging.getLogger("dd_pipeline.stages.lead_opt")


class PropertyPredictor:
    """Holds one model per endpoint and predicts them all for a SMILES list.

    This is the object the optimiser calls thousands of times, so it caches:
    an evolutionary loop revisits the same molecules constantly.
    """

    def __init__(self, featurizer: Featurizer, cfg: PipelineConfig) -> None:
        self.featurizer = featurizer
        self.cfg = cfg
        self.models: dict[str, EnsembleQSAR] = {}
        self.domains: dict[str, ApplicabilityDomain] = {}
        self._cache: dict[str, dict[str, float]] = {}

    def add(self, endpoint: str, model: EnsembleQSAR, ad: ApplicabilityDomain | None) -> None:
        self.models[endpoint] = model
        if ad is not None:
            self.domains[endpoint] = ad

    def predict_frame(self, smiles: list[str], include_physchem: bool = True) -> pd.DataFrame:
        uncached = [s for s in dict.fromkeys(smiles) if s not in self._cache]
        if uncached:
            fm = self.featurizer.transform(uncached)
            preds: dict[str, np.ndarray] = {}
            stds: dict[str, np.ndarray] = {}
            for ep, model in self.models.items():
                p = model.predict(fm.X)
                preds[ep] = p.mean
                stds[ep] = p.std
            n_fp = self.cfg.featurizer.fp_bits if self.cfg.featurizer.kind != "descriptors" else 0
            desc = fm.X[:, n_fp:] if n_fp else fm.X
            doms: dict[str, np.ndarray] = {}
            for ep, ad in self.domains.items():
                doms[ep] = ad.assess(uncached, desc)["in_domain"]

            for i, smi in enumerate(uncached):
                rec: dict[str, float] = {}
                for ep in self.models:
                    rec[ep] = float(preds[ep][i])
                    rec[f"{ep}_std"] = float(stds[ep][i])
                    if ep in doms:
                        rec[f"{ep}_in_domain"] = bool(doms[ep][i])
                if not fm.valid_mask[i]:
                    rec = {k: np.nan for k in rec}
                self._cache[smi] = rec

        rows = []
        for smi in smiles:
            rec = {"smiles": smi}
            rec.update(self._cache.get(smi, {}))
            if include_physchem:
                rec.update(_physchem(smi))
            rows.append(rec)
        out = pd.DataFrame(rows)
        cols = [f"{ep}_in_domain" for ep in self.domains if f"{ep}_in_domain" in out.columns]
        if cols:
            # In-domain for the overall molecule means in-domain for *every*
            # endpoint we are going to score it on.
            out["in_domain"] = out[cols].all(axis=1)
        else:
            # No domain model fitted. Do not claim coverage we cannot support:
            # leave it unknown rather than defaulting to True, which is how a
            # prediction on unrelated chemistry ends up on a shortlist.
            out["in_domain"] = pd.Series([pd.NA] * len(out), dtype="boolean")
        return out


# Thresholds for the domain-coverage guard below. Below `CRITICAL` an
# objective is effectively pure extrapolation; below `CAUTION` it is partly so.
DOMAIN_COVERAGE_CRITICAL = 0.05
DOMAIN_COVERAGE_CAUTION = 0.50

OUT_OF_DOMAIN_MESSAGE = (
    "objective %r is computed almost entirely outside its model's applicability "
    "domain (%.0f%% in domain). Those desirabilities are extrapolation, not "
    "prediction, so the MPO score is not meaningful for this objective. Either drop "
    "it, retrain that endpoint on chemistry resembling these molecules, or treat its "
    "contribution as unknown."
)


def check_objective_domain_coverage(
    frame: pd.DataFrame,
    objectives: list,
    label: str = "molecules",
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """Per objective, the fraction of molecules its own model was trained to
    speak about. Returns {endpoint: fraction_in_domain}.

    This catches a failure the MPO is otherwise silent about. Desirability
    treats a *missing* prediction as zero, but an out-of-domain prediction is
    not missing -- it is a confident-looking extrapolation, and it gets scored
    like any other number. When endpoints come from different datasets covering
    different chemistry (the normal case: potency from the project's own series,
    ADMET from public data), an objective can be evaluated entirely outside its
    model's domain while the combined score looks perfectly healthy.

    Found for real on BACE-1 inhibitors scored against an ESOL-trained
    solubility model: 0% domain coverage, and the MPO reported nothing wrong.
    """
    log = logger or LOG
    coverage: dict[str, float] = {}
    for obj in objectives:
        col = f"{obj.endpoint}_in_domain"
        if col not in frame.columns:
            continue
        frac = float(frame[col].fillna(False).astype(bool).mean()) if len(frame) else 0.0
        coverage[obj.endpoint] = frac
        if frac < DOMAIN_COVERAGE_CRITICAL:
            log.error("%s: " + OUT_OF_DOMAIN_MESSAGE, label, obj.endpoint, 100 * frac)
        elif frac < DOMAIN_COVERAGE_CAUTION:
            log.warning(
                "%s: objective %r is in-domain for only %.0f%% of molecules; treat its "
                "contribution to the MPO score with caution.",
                label, obj.endpoint, 100 * frac,
            )
    return coverage


def _physchem(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {"mw": np.nan, "clogp": np.nan, "tpsa": np.nan, "hbd": np.nan,
                "heavy_atoms": np.nan, "sa_score": 10.0, "qed": np.nan}
    from rdkit.Chem import Descriptors
    return {
        "mw": float(rdMolDescriptors.CalcExactMolWt(mol)),
        "clogp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "sa_score": float(sa_score_smiles(smiles)),
        "qed": float(Descriptors.qed(mol)),
    }


def train_admet_models(
    assay: pd.DataFrame,
    endpoints: list[str],
    cfg: PipelineConfig,
    run: Run,
    featurizer: Featurizer,
    smiles_col: str = "canonical_smiles",
) -> tuple[PropertyPredictor, dict[str, dict]]:
    LOG.info("--- training %d ADMET models ---", len(endpoints))
    predictor = PropertyPredictor(featurizer, cfg)
    all_metrics: dict[str, dict] = {}

    for ep in endpoints:
        if ep not in assay.columns:
            LOG.warning("endpoint %s absent from the assay table; skipped", ep)
            continue
        data = assay.dropna(subset=[ep]).reset_index(drop=True)
        if len(data) < 60:
            LOG.warning("endpoint %s has only %d points; skipped", ep, len(data))
            continue

        # Clearance is a rate: model it on a log scale or the loss is
        # dominated by the fast-turnover compounds.
        target = data[ep].to_numpy(dtype=float)
        log_transformed = ep in ("hlm_clint", "clint", "rlm_clint")
        if log_transformed:
            target = np.log10(np.clip(target, 1e-3, None))

        tr, va, te = make_split(data, cfg.split, smiles_col=smiles_col)
        if len(va) < 20:
            n_take = min(max(20 - len(va), 0), max(len(tr) - 40, 0))
            va = np.concatenate([va, tr[:n_take]])
            tr = tr[n_take:]

        fm = featurizer.transform(data[smiles_col].tolist())
        model = EnsembleQSAR(cfg.admet_model, endpoint=ep)
        model.fit(fm.X[tr], target[tr], fm.X[va], target[va],
                  feature_names=fm.feature_names)

        ad = ApplicabilityDomain(cfg.applicability, featurizer)
        n_fp = cfg.featurizer.fp_bits if cfg.featurizer.kind != "descriptors" else 0
        ad.fit(data.loc[tr, smiles_col].tolist(),
               fm.X[tr][:, n_fp:] if n_fp else fm.X[tr])

        pred = model.predict(fm.X[te])
        m = evaluate_regression(target[te], pred)
        fps = featurizer.bit_fingerprints(data[smiles_col].tolist())
        m.update(baseline_metrics(target[tr], target[te],
                                  [fps[i] for i in tr], [fps[i] for i in te]))
        m["log_transformed"] = bool(log_transformed)

        model.card.n_test = int(len(te))
        model.card.split_strategy = cfg.split.strategy
        model.card.featurizer = cfg.featurizer.model_dump()
        model.card.metrics = {k: v for k, v in m.items() if isinstance(v, (int, float))}
        if log_transformed:
            model.card.notes = "Target was log10-transformed; predictions are log10 units."
        model.save(run.path("models", f"admet_{ep}.pkl"))

        predictor.add(ep, model, ad)
        all_metrics[ep] = m
        run.log_metrics(f"admet_{ep}", m)
        LOG.info("  %-12s n=%4d  RMSE=%.3f  Spearman=%+.3f  (1-NN %.3f)",
                 ep, len(data), m["rmse"], m["spearman"],
                 m.get("baseline_1nn_rmse", float("nan")))

    run.save_json(all_metrics, "admet_model_metrics", "lead_opt")
    return predictor, all_metrics


def _back_transform(df: pd.DataFrame, predictor: PropertyPredictor) -> pd.DataFrame:
    """Undo the log10 on clearance so MPO bounds are in the units a chemist
    wrote them in (uL/min/mg), not log units."""
    out = df.copy()
    for ep, model in predictor.models.items():
        if "log" in (model.card.notes or "") and ep in out.columns:
            out[ep] = 10 ** out[ep].to_numpy(dtype=float)
    return out


def optimize_leads(
    hits: pd.DataFrame,
    predictor: PropertyPredictor,
    cfg: PipelineConfig,
    run: Run,
    smiles_col: str = "canonical_smiles",
) -> dict[str, pd.DataFrame]:
    LOG.info("=== Stage 3: multi-objective lead optimisation ===")
    scorer = MPOScorer(cfg.lead_opt)
    alerts = StructuralAlertFilter(cfg.hit_discovery.apply_pains, cfg.hit_discovery.apply_brenk)

    def filter_fn(smi: str) -> bool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        return not alerts.check(mol) and not property_gate(mol, cfg.hit_discovery)

    beta = cfg.lead_opt.uncertainty_penalty

    def _conservative(frame: pd.DataFrame) -> pd.DataFrame:
        """Shift predictions against their preferred direction before scoring.

        Without this the optimiser is rewarded for finding the places where the
        model is most wrong, since that is where predictions look best.
        """
        return pessimistic_frame(frame, cfg.lead_opt.objectives, beta=beta)

    def score_fn(smiles: list[str]) -> np.ndarray:
        frame = _back_transform(predictor.predict_frame(smiles), predictor)
        return scorer.score_frame(_conservative(frame)).score

    if beta > 0:
        LOG.info("scoring on uncertainty-penalised predictions (beta=%.2f sigma)", beta)

    def _check_domain_coverage(frame: pd.DataFrame, label: str) -> dict[str, float]:
        return check_objective_domain_coverage(frame, cfg.lead_opt.objectives, label)

    # --- score the starting hits -----------------------------------------
    seed_smiles = hits[smiles_col].astype(str).tolist()[: cfg.lead_opt.n_seeds]
    seed_frame = _back_transform(predictor.predict_frame(seed_smiles), predictor)
    seed_scored = scorer.annotate(_conservative(seed_frame))
    # Keep the optimistic score alongside it: the gap between the two is a
    # direct readout of how much of a molecule's appeal is model uncertainty.
    seed_scored["mpo_score_optimistic"] = scorer.score_frame(seed_frame).score
    seed_scored["origin"] = "hit"
    seed_scored["generation"] = 0
    seed_coverage = _check_domain_coverage(seed_scored, "starting hits")
    run.save_json(seed_coverage, "objective_domain_coverage", "lead_opt")
    run.save_table(seed_scored, "lead_opt_seeds", "lead_opt")
    LOG.info("seed hits: MPO mean=%.3f max=%.3f",
             seed_scored["mpo_score"].mean(), seed_scored["mpo_score"].max())

    # --- evolve -----------------------------------------------------------
    ga = GeneticOptimizer(
        score_fn=score_fn,
        config=GAConfig(
            population_size=max(cfg.lead_opt.n_analogs_per_seed, 100),
            n_generations=cfg.lead_opt.max_generations,
            max_sa_score=cfg.lead_opt.max_sa_score,
            min_similarity_to_seed=cfg.lead_opt.min_similarity_to_seed,
            seed=cfg.seed,
        ),
        filter_fn=filter_fn,
    )
    analogs = ga.run(seed_smiles)
    run.save_json(ga.history, "lead_opt_ga_history", "lead_opt")
    LOG.info("generated %d admissible analogs", len(analogs))

    if not analogs:
        LOG.warning("no analogs survived the gates; returning the seed hits only")
        combined = seed_scored
    else:
        an_frame = _back_transform(
            predictor.predict_frame([a.smiles for a in analogs]), predictor
        )
        an_scored = scorer.annotate(_conservative(an_frame))
        an_scored["mpo_score_optimistic"] = scorer.score_frame(an_frame).score
        an_scored["origin"] = [a.origin for a in analogs]
        an_scored["generation"] = [a.generation for a in analogs]
        an_scored["parent"] = [a.parent for a in analogs]
        an_scored["similarity_to_parent"] = [a.similarity_to_parent for a in analogs]
        run.save_table(an_scored, "lead_opt_analogs", "lead_opt")
        combined = pd.concat([seed_scored, an_scored], ignore_index=True)

    # --- efficiency metrics ----------------------------------------------
    if "pIC50" in combined.columns:
        combined["ligand_efficiency"] = ligand_efficiency(
            combined["pIC50"].to_numpy(), combined["heavy_atoms"].to_numpy()
        )
        combined["lipophilic_efficiency"] = lipophilic_efficiency(
            combined["pIC50"].to_numpy(), combined["clogp"].to_numpy()
        )

    # --- Pareto analysis --------------------------------------------------
    # Run the front over the *desirability* columns, not the raw endpoints.
    # Raw values need a per-objective direction, and a `window` objective has
    # no single direction at all -- logD 2.0 beats both 0.5 and 6.0, so
    # treating it as "higher is better" silently rewards over-lipophilic
    # compounds. Desirabilities are all on 0-1 and all maximised, so the
    # comparison is well defined for every objective kind.
    obj_cols = [f"d_{o.endpoint}" for o in cfg.lead_opt.objectives
                if f"d_{o.endpoint}" in combined.columns]
    maximize = [True] * len(obj_cols)
    if not obj_cols:
        LOG.warning("no desirability columns present; skipping Pareto analysis")
        obj_cols = []
    valid = (combined[obj_cols].notna().all(axis=1) if obj_cols
             else pd.Series(False, index=combined.index))
    combined["pareto_rank"] = np.nan
    combined["on_pareto_front"] = False
    combined["crowding_distance"] = np.nan
    if valid.sum() > 2:
        sub = combined.loc[valid, obj_cols].to_numpy(dtype=float)
        combined.loc[valid, "pareto_rank"] = pareto_ranks(sub, maximize)
        front = pareto_front(sub, maximize)
        combined.loc[valid, "on_pareto_front"] = front
        front_idx = combined.loc[valid].index[front]
        if len(front_idx) > 2:
            combined.loc[front_idx, "crowding_distance"] = crowding_distance(
                combined.loc[front_idx, obj_cols].to_numpy(dtype=float)
            )
        front_frac = float(front.sum() / max(valid.sum(), 1))
        LOG.info("Pareto front: %d of %d scorable molecules (%.0f%%) over %d objectives",
                 int(front.sum()), int(valid.sum()), 100 * front_frac, len(obj_cols))
        if front_frac > 0.4 and len(obj_cols) > 3:
            # With many objectives almost nothing is dominated, so the front
            # stops discriminating. This is the curse of dimensionality, not a
            # bug -- but the front is then not a useful shortlist on its own.
            LOG.warning(
                "%.0f%% of molecules are non-dominated across %d objectives; the Pareto "
                "front is not discriminating here. Rank on mpo_score, or re-run the "
                "front over the 2-3 objectives the project is actually trading off.",
                100 * front_frac, len(obj_cols),
            )
        # A 2-3 objective front is what a project team can actually read.
        key_cols = [c for c in ("d_pIC50", "d_hlm_clint", "d_herg_pic50") if c in obj_cols]
        if len(key_cols) >= 2:
            key_max = [True] * len(key_cols)
            sub_k = combined.loc[valid, key_cols].to_numpy(dtype=float)
            combined["on_key_pareto_front"] = False
            combined.loc[valid, "on_key_pareto_front"] = pareto_front(sub_k, key_max)
            LOG.info("key-objective front (%s): %d molecules",
                     "/".join(key_cols), int(combined["on_key_pareto_front"].sum()))

    combined["scaffold"] = [murcko_scaffold(s) for s in combined["smiles"]]
    run.save_table(combined, "lead_opt_all_scored", "lead_opt")

    # --- final candidate selection ---------------------------------------
    pool = combined[
        (~combined["fails_critical"])
        & (combined["sa_score"] <= cfg.lead_opt.max_sa_score)
    ].copy()
    if "in_domain" in pool.columns:
        in_dom = pool[pool["in_domain"].fillna(False).astype(bool)]
        if len(in_dom) >= cfg.lead_opt.n_final_candidates:
            pool = in_dom
        else:
            LOG.warning("only %d in-domain candidates; keeping out-of-domain ones flagged",
                        len(in_dom))

    featurizer = predictor.featurizer
    if pool.empty:
        LOG.warning("no candidate cleared the critical objectives")
        candidates = pool
    else:
        # One per scaffold first, so the shortlist is a set of series, not a
        # set of near-identical molecules.
        pool = pool.sort_values("mpo_score", ascending=False)
        per_scaffold = pool.groupby("scaffold", sort=False).head(2)
        from ..scoring.pareto import diverse_pick

        fps = featurizer.bit_fingerprints(per_scaffold["smiles"].tolist())
        pick = diverse_pick(
            fps, per_scaffold["mpo_score"].to_numpy(dtype=float),
            n=cfg.lead_opt.n_final_candidates, max_similarity=0.85,
        )
        candidates = per_scaffold.iloc[pick].reset_index(drop=True)
        candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)

    run.save_table(candidates, "final_candidates", "lead_opt")

    # Report best AND mean, because they answer different questions. The best
    # molecule often remains a starting hit (elitism preserves it), while the
    # whole population improves -- that is a real win, since the shortlist is
    # 25 compounds, not one. A rise in the mean with a flat max means the
    # optimiser is broadening a viable series rather than finding a new peak.
    best_seed = float(seed_scored["mpo_score"].max()) if len(seed_scored) else float("nan")
    mean_seed = float(seed_scored["mpo_score"].mean()) if len(seed_scored) else float("nan")
    best_final = float(candidates["mpo_score"].max()) if len(candidates) else float("nan")
    mean_final = float(candidates["mpo_score"].mean()) if len(candidates) else float("nan")
    best_analog = float(
        combined.loc[combined["origin"] != "hit", "mpo_score"].max()
    ) if (combined["origin"] != "hit").any() else float("nan")
    n_analogs_beating_best_seed = int(
        ((combined["origin"] != "hit") & (combined["mpo_score"] > best_seed)).sum()
    )

    run.log_metrics("lead_opt", {
        "n_seeds": int(len(seed_scored)),
        "n_analogs_generated": int(len(analogs)),
        "n_scored_total": int(len(combined)),
        "n_pareto_front": int(combined["on_pareto_front"].sum()),
        "n_final_candidates": int(len(candidates)),
        "best_seed_mpo": best_seed,
        "mean_seed_mpo": mean_seed,
        "best_analog_mpo": best_analog,
        "best_candidate_mpo": best_final,
        "mean_candidate_mpo": mean_final,
        "mpo_improvement_best": best_final - best_seed,
        "mpo_improvement_mean": mean_final - mean_seed,
        "n_analogs_beating_best_seed": n_analogs_beating_best_seed,
        "n_unique_scaffolds_final": int(candidates["scaffold"].nunique()) if len(candidates) else 0,
        "uncertainty_penalty_beta": float(beta),
        **{f"domain_coverage_{k}": v for k, v in seed_coverage.items()},
        "min_objective_domain_coverage": (
            float(min(seed_coverage.values())) if seed_coverage else None
        ),
        # How much of the shortlist's apparent quality is model uncertainty.
        "mean_optimism_gap": float(
            (candidates["mpo_score_optimistic"] - candidates["mpo_score"]).mean()
        ) if len(candidates) and "mpo_score_optimistic" in candidates else None,
    })
    LOG.info("MPO best: seed %.3f -> candidate %.3f (%+.3f); "
             "mean: %.3f -> %.3f (%+.3f); %d analogs beat the best starting hit",
             best_seed, best_final, best_final - best_seed,
             mean_seed, mean_final, mean_final - mean_seed, n_analogs_beating_best_seed)
    if n_analogs_beating_best_seed == 0 and len(analogs):
        LOG.warning(
            "no generated analog beat the best starting hit. The hits may already sit at "
            "a local optimum for these objectives; widen the replacement rules, allow "
            "lower min_similarity_to_seed, or accept that the series needs new chemistry."
        )

    return {"seeds": seed_scored, "all_scored": combined, "candidates": candidates}
