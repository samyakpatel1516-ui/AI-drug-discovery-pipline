# dd_pipeline — AI-based drug discovery pipeline

Target identification → hit discovery → lead optimisation, as runnable code
with provenance, honest validation and calibrated uncertainty.

> **Status and provenance.** This codebase was generated with AI assistance
> (Claude) in a single working session, including the validation experiments in
> `REAL_DATA.md`. It is retrospectively benchmarked, not validated: see
> [Validation status](#validation-status) for exactly what has and has not been
> established, stage by stage. Nothing here has been tested prospectively, and
> no compound has been made.

```
omics + annotations          assay data              hits + ADMET models
        │                        │                          │
        ▼                        ▼                          ▼
┌───────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│ Stage 1       │      │ Stage 2          │      │ Stage 3             │
│ target ID     │ ───▶ │ hit discovery    │ ───▶ │ lead optimisation   │
│               │      │                  │      │                     │
│ • DE + FDR    │      │ • standardise    │      │ • ADMET models      │
│ • dependency  │      │ • scaffold split │      │ • MPO desirability  │
│ • druggability│      │ • QSAR ensemble  │      │ • GA analog design  │
│ • safety      │      │ • conformal PI   │      │ • Pareto front      │
│ • modality    │      │ • alerts/filters │      │ • diverse shortlist │
└───────────────┘      └──────────────────┘      └─────────────────────┘
        │                        │                          │
        └────────────────────────┴──────────────────────────┘
                                 ▼
                 run directory: tables, models, model cards,
                 metrics, config, env, manifest with hashes
```

## Install and run

```bash
pip install -r requirements.txt

# Validation on real published data (BACE-1, logD, solubility)
PYTHONPATH=. python examples/real_data_bace.py

# Full pipeline on simulated data — runs in a few minutes, writes runs/<project>_<id>/
python -m dd_pipeline demo

# Your own data
python -m dd_pipeline config-template > configs/program.yaml
python -m dd_pipeline run --config configs/program.yaml \
    --assay data/potency.csv --library data/vendor.smi \
    --admet logS=data/sol.csv hlm_clint=data/clint.csv

# Score a library with a model you trained earlier
python -m dd_pipeline screen --model runs/.../models/activity_pIC50.pkl \
    --library data/vendor.smi --out hits.csv

# Markdown summary of any completed run
python -m dd_pipeline report runs/demo-program_full

pytest -m "not slow"    # 74 fast tests
pytest -m slow          # full-pipeline integration test
```

Two worked examples:

- `examples/custom_program.py` — the template to copy. Four marked decisions:
  validation strategy, the target product profile as desirability objectives,
  screening economics, and your data loaders.
- `examples/active_learning_demo.py` — compares four compound-selection
  policies over five design-make-test rounds.

`--assay` wants `smiles,value,units[,relation]`. Units are normalised to
p-scale, and qualified values (`>`, `<`) are dropped rather than treated as
exact measurements.

## What is worth your attention

Most of the machinery here is ordinary. These are the parts that determine
whether the output is trustworthy, and they are where to look first when
adapting this to your program.

**Scaffold splitting is the default, not random splitting.**
`chem/splits.py`. A random split over a congeneric series measures
interpolation between analogs, which is not what a virtual screen does. The
same model typically loses 0.3–0.6 log units of RMSE moving from a random to a
scaffold split; that gap is information about your model, not a problem with
the split. `temporal_split` is closer still to prospective use and usually
looks worst — report it to decision makers anyway.

**Enrichment is measured against the pool the model actually ranked.**
This denominator is easy to get wrong and the error is invisible. An earlier
version of the retrospective check generated a *separate* 2000-compound
library as the "random pick" baseline. That is not the same distribution —
`build_virtual_library` keeps the first N products passing the lead-like gate,
so an n=2000 draw differs systematically from an n=6000 one. The bogus
baseline put 14% of compounds at pIC50 ≥ 7 where the real screened pool had
1.3%, and reported a screening enrichment of **0.92** — apparently no better
than random — while the true figure was about **10×**.

Measured correctly against the screened pool (filters applied, training
compounds removed):

| selection (top 200) | hit rate ≥6 | EF ≥6 | hit rate ≥7 | EF ≥7 |
|---|---|---|---|---|
| whole screened pool (n=2433) | 0.324 | 1.0 | 0.013 | 1.0 |
| greedy by predicted activity | 1.000 | 3.08 | 0.155 | **11.8** |
| greedy by lower confidence bound | 1.000 | 3.08 | 0.150 | 11.4 |
| delivered hit list (LCB + diversity cap) | 0.995 | 3.07 | 0.130 | **9.9** |

The last row is what the pipeline hands you, and the comparison prices the
diversity cap explicitly: it costs about 16% of top-end enrichment
(11.8 → 9.9) and buys 151 unique scaffolds in 200 compounds. That is a trade
worth making for a first screen and a bad one if you are confirming a known
series — which is why `max_pairwise_similarity` is in the config and not
hard-coded.

**Every model is reported against a 1-nearest-neighbour baseline.**
`models/evaluate.py`. A QSAR model that cannot beat 1-NN on fingerprints has
learned nothing beyond "similar molecules have similar activity" and does not
justify its maintenance cost. This is the first number to check.

**Uncertainty is conformally calibrated, and the calibration is tested.**
`models/qsar.py`. Ensemble spread is not a probability. Split conformal
calibration on a held-out fold gives intervals with a coverage guarantee and
no distributional assumptions. `conformal_coverage_90` in the metrics should
land near 0.90; if it does not, the intervals are lying and nothing downstream
that uses them is safe. The nonconformity score is normalised by the median
calibration spread rather than a bare epsilon — without that, compounds whose
ensemble members happen to agree drive the quantile to absurd values.

**Ensemble members get jittered hyperparameters, not just different seeds.**
Members trained identically on bootstrap samples agree almost everywhere, and
their spread is then useless as an uncertainty signal. The failure mode that
matters is a model that is confidently wrong on novel chemistry, which is
exactly where the warning is needed.

**Applicability domain gates every prediction.**
`models/applicability.py`. A model returns a confident-looking number for any
molecule. Predictions below the similarity threshold to the training set are
flagged `in_domain=False` and must not drive purchase or synthesis decisions.

**MPO uses a weighted geometric mean, with hard gates on critical endpoints.**
`scoring/desirability.py`. Arithmetic averaging lets a compound excel its way
past a fatal flaw. Geometric does not: a zero on any term collapses the score.
Potency alone reliably optimises toward greasy, insoluble, hERG-active
molecules — the point of the MPO is to make that impossible rather than
merely discouraged.

**Compound selection splits its budget between exploitation and exploration.**
`scoring/pareto.py`. High-scoring in-domain compounds are the ones you expect
to work; high-uncertainty in-domain compounds are the ones that teach the model
most per assay well. A batch of pure exploitation confirms what you already
believe and makes the next iteration no better. Selection also applies a
Tanimoto cap so you do not buy twenty analogs of one compound.

**The loop is a first-class stage, not an afterthought.**
`stages/active_learning.py`. A programme is select → measure → retrain →
select, and the number that matters is improvement per assay plate, not static
model quality. `compare_policies` runs the same loop under random, greedy, UCB
and diversity-capped selection from identical starting data, so you can pick a
screening cadence on evidence. `oracle_fn` is the seam: here it is the
simulator, in production it is the function that reads this week's results out
of your LIMS. On the simulated landscape the model-driven policies clearly beat
random (hit rate 1.00 vs 0.76) while greedy and diverse-UCB roughly tie —
which is itself the point. Run it on your data instead of assuming.

**Structure standardisation happens once, before anything else.**
`chem/standardize.py`. Salt forms, charge states and tautomers of the same
compound become separate training rows, and duplicates then leak across the
train/test boundary. Replicates are collapsed to a median, and their spread is
kept — it is the floor on the accuracy any model can reach, reported as
`rmse_over_assay_noise`. Below about 1.0 you are at the noise floor and more
data will not help.

**Generative design is scored on uncertainty-penalised predictions.**
`scoring/desirability.pessimistic_frame`, controlled by
`lead_opt.uncertainty_penalty` (β, default 1.0). An optimiser scored on
predicted means marches straight into the region where the model is most
wrong, because that is where the predictions look best. Each endpoint is
therefore shifted β standard deviations *against* its preferred direction
before scoring, so a molecule can only win by looking good somewhere the model
is also confident.

**On the simulated landscape, the penalty is insurance that is not currently
paying out, and the honest thing is to say so.** A controlled β=1 vs β=0 run
on identical data:

| | β=0 | β=1 |
|---|---|---|
| predicted vs true MPO ρ on the shortlist | **0.933** | 0.863 |
| mean *true* MPO of the shortlist | 0.3278 | 0.3277 |
| mean predicted MPO of the shortlist | 0.365 | 0.318 |
| analogs beating the best starting hit | 3 | 6 |

β=0 shows no over-optimisation here and slightly *better* ranking fidelity;
the true quality of the shortlist is identical to four decimal places. The
penalty's only visible effect is to make the predicted scores less optimistic
(`mean_optimism_gap` 0.044). Keep it on as cheap insurance for a real
programme, where models are worse and the optimiser pushes harder than a
3-generation rule-based GA — but do not expect it to improve anything, and
turn it off if it is costing you.

An earlier run did report ρ = **−0.11**, and chasing that number is how two
of the bugs below were found. It was not over-optimisation: the diagnostic
was slicing arrays instead of joining on SMILES, so it compared different
molecules, and the library that run drew was pathological. Both are fixed. The
lesson is about the diagnostic, not the optimiser: a monitoring metric with a
bug in it will invent a crisis, and the only reason this resolved correctly is
that the simulated oracle allowed a controlled comparison.

**Generative design stays in synthesisable space.**
`chem/enumerate.py`. Analogs come from medicinal-chemistry replacement rules
and BRICS fragment recombination, scored against the full MPO, gated on
synthetic accessibility and structural alerts every generation. This gives up
the reach of a latent-space generator in exchange for proposals a chemist can
actually make.

## A reproducibility bug worth knowing about

This one is general, not specific to chemistry, and it defeated the test suite
that was written to catch it.

`BRICS.BRICSDecompose` returns a **set** of fragment SMILES. Iteration order
over a set of *strings* depends on per-process hash randomisation, so the
fragment pool — and therefore every generated library — differed on each
invocation, with identical seeds and identical config. The in-process test
(`build twice, compare`) passed happily, because both calls shared one hash
seed. The manifests recorded a seed that did not determine the output, which
makes the provenance guarantee worthless in precisely the situation it exists
for: reproducing a result months later.

Two consequences worth generalising:

- `os.environ["PYTHONHASHSEED"] = str(seed)` at runtime does nothing. It is
  read at interpreter start. Setting it inside `set_global_seed` was
  actively misleading, so it is gone; code that must be reproducible avoids
  depending on string set/dict ordering instead.
- Reproducibility tests have to cross a process boundary.
  `test_library_is_reproducible_across_processes` runs the generator under
  three `PYTHONHASHSEED` values and compares digests.

The fix is `sorted(frags)`. The symptom, before anyone noticed the cause, was
one library draw where 5980 of 6000 compounds tripped structural alerts and
the hit list collapsed to two compounds.

## Validation on real data

`REAL_DATA.md` has the full write-up; `examples/real_data_bace.py` reproduces
it. Three published datasets with measured values — BACE-1 pIC50 (1500
compounds), logD7.4 (4175), aqueous solubility (1011) — all MoleculeNet
benchmarks.

Headline results:

- **Competitive with published graph networks.** Scaffold-split RMSE 0.903 on
  ESOL (D-MPNN: 0.91) and 0.749 on Lipophilicity (D-MPNN: 0.64), ahead of
  ChemBERTa and GROVER on both. Counted ECFP + XGBoost, trained in seconds.
  Indicative rather than head-to-head — split protocols differ and my ESOL set
  excludes 106 sub-fragment molecules.
- **Retrospective virtual screen on BACE-1**, scaffold-disjoint holdout diluted
  with property-matched decoys: EF@1% = **15.8** at pIC50 ≥ 7 (ceiling 19.4)
  and **38.6** at pIC50 ≥ 8. The delivered hit list enriches 9.7×, the
  difference being the price of the diversity cap.
- **Conformal coverage 0.85–0.92** for nominal 0.90, degrading systematically
  on scaffold splits.

And the finding that justifies the applicability-domain machinery existing:
scoring real BACE-1 inhibitors against an ESOL-trained solubility model gives
**0% domain coverage** — large peptidomimetics versus small substituted
benzenes. The MPO combined it with potency and reported nothing wrong, because
desirability scores a *missing* prediction as zero but an out-of-domain
prediction is a confident extrapolation that looks like any other number.
`check_objective_domain_coverage` now raises an ERROR below 5% coverage. When
endpoints come from different datasets, an MPO is only as meaningful as the
chemical-space overlap between them.

## Validation status

"Validated" is a strong word in drug discovery and it does not apply here.
What exists is software testing plus retrospective benchmarking.

| Level | Status |
|---|---|
| Software correctness — the code does what it says | **Yes** — 84 tests, including regression tests for five real bugs |
| Retrospective benchmarking — beats baselines on held-out public data | **Yes**, for the QSAR models (3 endpoints, scaffold split, vs 1-NN and constant baselines, compared to published numbers) |
| Statistical validation — repeated splits, confidence intervals, y-randomisation | **No** |
| Prospective validation — predictions tested by experiment | **No** |
| Formal validation — OECD QSAR principles, GxP, CSV | **No** |

Specific weaknesses, so nobody has to find them by reading the code:

- **Everything is retrospective.** Measured values were withheld until after
  prediction, which simulates prospective use, but the compounds already
  existed and were chosen by chemists who had SAR knowledge. A real screening
  deck is not assembled that way.
- **One split, one seed, no error bars.** A single deterministic Bemis-Murcko
  split per dataset. RMSE 0.903 has no confidence interval attached.
- **No y-randomisation.** Shuffling labels to confirm performance collapses is
  a standard QSAR requirement and has not been done.
- Against OECD's five principles: defined endpoint ✓, unambiguous algorithm ✓,
  defined applicability domain ✓, appropriate measures of fit/robustness/
  predictivity — partial (no robustness testing), mechanistic
  interpretation ✗.

**Validation is uneven across the three stages**, which matters because the
headline numbers all come from one of them:

- **Stage 2 (QSAR + virtual screening)** — reasonably evidenced on real data.
- **Stage 1 (target identification)** — **no real-data validation at all.** It
  recovers 12/12 planted targets, but the targets were planted here. That
  tests the arithmetic, not the biology. Validating it needs real TCGA/DepMap
  data and a check against targets with known clinical outcomes.
- **Stage 3 (MPO + generative design)** — the two ADMET models are
  benchmarked; the MPO and analog search are not. On real data one of three
  objectives turned out to be pure extrapolation (see above), so the reported
  MPO improvement was partly against a meaningless score.

Cheapest things that would materially strengthen this: repeat every split
across 5–10 seeds for confidence intervals, add y-randomisation, and add a
time-split if dated assay data is available. The only thing that settles it is
making compounds and running the assay.

## Honest limitations

- **The demo data is simulated.** `data/simulate.py` generates endpoints from
  descriptor-and-substructure rules. It is built to be *hard* in the right ways
  — activity cliffs, correlated liabilities, biased training-set selection,
  realistic assay noise, pan-essential and undruggable decoys among the planted
  targets — but metrics from it describe the pipeline, not chemistry. Swap in
  `data/loaders.py` against your own data before concluding anything
  scientific. The `oracle_retrospective` block in the manifest exists only
  because the truth is known; you will not have it.
- **No structure-based scoring.** There is no docking, no free-energy
  perturbation, no cryptic-pocket analysis. Ligand-based QSAR is the right
  first tool when you have measured compounds and the wrong one when you have
  a structure and no chemical matter. FEP belongs between stage 2 and stage 3
  on a few dozen compounds, not in a screening loop.
- **Pareto fronts stop discriminating above three or four objectives.** With
  six, most molecules are non-dominated; the code warns when this happens and
  also computes a front over potency/clearance/hERG alone. Rank on `mpo_score`
  and use the front to show trade-offs, not to shortlist.
- **The ADMET endpoints are the cheap ones.** Real programs also need
  permeability in the relevant cell line, plasma protein binding, CYP
  inhibition and induction, transporter liabilities, and in vivo PK. Each is a
  new `ObjectiveSpec` and a new model — the machinery extends, the biology does
  not get easier.
- **Target identification is a triage tool.** It ranks a few thousand genes
  down to a shortlist worth a structural biologist's time. It is not a
  substitute for that opinion, and the druggability priors are crude.
- **Single-task models.** With correlated endpoints and sparse measurements, a
  multi-task model usually wins. The `EnsembleQSAR` interface is the seam to
  replace.

## Replacing pieces

Everything is behind a narrow interface, so these are local changes:

| Swap | Where | Note |
|---|---|---|
| Learned representation (GNN, pretrained transformer) | `chem/featurize.py` — keep the `Featurizer` interface | Rarely beats counted ECFP + descriptors below ~10⁴ compounds per endpoint |
| Multi-task / transfer learning | `models/qsar.py` — `EnsembleQSAR` | The main modelling upgrade worth making |
| Docking or FEP scores | new column, then an `ObjectiveSpec` | Treat as another objective, not as truth |
| Reinforcement-learning or latent-space generator | `chem/enumerate.py` — keep `score_fn` | Gate hard on SA score and alerts, or it will exploit model error |
| Real omics (DESeq2/limma output) | `targets/omics.py` — `differential_expression` accepts precomputed stats | Run proper count-based models upstream |
| Bayesian optimisation for batch selection | `scoring/pareto.py` — `select_batch` | UCB is a one-line change to the exploration ranking |

## Run directory layout

```
runs/<project>_<id>/
  config.yaml              exact config used
  environment.json         python, platform, git commit, package versions
  manifest.json            config + metrics + artifact hashes
  data/*.parquet           every intermediate table
  models/*.pkl             fitted ensembles
  models/*.card.json       model cards: n_train, split, metrics, assay noise
  reports/*.json           metrics, feature importances, GA history
  logs/pipeline.log
```

A candidate you cannot trace to a model, a config and a data hash is not
auditable. The manifest is what makes a shortlist defensible six months later.
