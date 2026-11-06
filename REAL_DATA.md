# Validation on real measured data

No simulation anywhere in this document. Every number comes from published
experimental measurements.

Reproduce with:

```bash
PYTHONPATH=. python examples/real_data_bace.py
```

## The data

| Endpoint | Dataset | n (after curation) | What it measures |
|---|---|---|---|
| `pIC50` | BACE-1 | 1500 | Human β-secretase 1 inhibition, a real Alzheimer's target |
| `logD` | Lipophilicity | 4175 | Experimental octanol/water distribution at pH 7.4 |
| `logS` | ESOL / Delaney | 1011 | Measured aqueous solubility, log₁₀ mol/L |

All three are MoleculeNet benchmark sets ([Wu et al., *Chem. Sci.* 2018](https://pubs.rsc.org/sc/article/9/2/513/584204/MoleculeNet-a-benchmark-for-molecular-machine)).
BACE-1 data originates from [Subramanian et al., *J. Chem. Inf. Model.* 2016](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00290);
solubility from Delaney, *J. Chem. Inf. Comput. Sci.* 2004. Loaders, source
URLs and SHA-256 hashes are in `dd_pipeline/data/public.py`; provenance for
each file is written into the run directory.

**Curation found real problems in these published files.** The standardiser
collapsed 9 duplicate structure groups in BACE, 19 in Lipophilicity and 11 in
ESOL. Those are the same compound under different SMILES spellings, salt forms
or tautomers. Left alone, they leak across the train/test boundary and inflate
every metric below. It also rejected 4, 3 and 106 unparsable or out-of-scope
structures respectively — and that 106 is itself a finding (see the ESOL
caveat).

## Experiment 1 — how much does the split protocol matter?

Identical model and features; only the split changes.

| Dataset | Endpoint | n | Split | RMSE | R² | Spearman ρ | 1-NN RMSE | Conformal coverage |
|---|---|---|---|---|---|---|---|---|
| BACE-1 | pIC50 | 1500 | random | 0.746 | 0.692 | 0.808 | 1.007 | 0.913 |
| BACE-1 | pIC50 | 1500 | **scaffold** | 0.658 | **0.628** | **0.768** | 0.824 | 0.873 |
| Lipophilicity | logD | 4175 | random | 0.675 | 0.689 | 0.810 | 1.023 | 0.920 |
| Lipophilicity | logD | 4175 | **scaffold** | **0.749** | 0.653 | 0.816 | 1.353 | 0.848 |
| ESOL | logS | 1011 | random | 0.658 | 0.883 | 0.938 | 1.648 | 0.916 |
| ESOL | logS | 1011 | **scaffold** | **0.903** | 0.847 | 0.947 | 2.423 | 0.896 |

Every model beats both a constant predictor and 1-nearest-neighbour on
fingerprints, on both splits. Conformal 90% intervals cover 0.85–0.92 of
truths, so the uncertainty is roughly honest; note it degrades systematically
on scaffold splits (0.873, 0.848, 0.896 vs 0.913, 0.920, 0.916), which is the
calibration set being less representative of novel chemistry.

**An important methodological catch.** ESOL and Lipophilicity behave as
expected — scaffold RMSE is worse (+0.245 and +0.074). BACE *appears to
improve* (0.746 → 0.658), which is an artifact: RMSE is not comparable across
different test sets, and the scaffold test set here has lower variance. Judge
the degradation on R² and Spearman, which are variance-normalised and do move
the right way (0.692 → 0.628, 0.808 → 0.768). The 1-NN baseline shifts the
same way (1.007 → 0.824), confirming the test set got *easier in spread* while
the model's relative skill dropped. Quoting RMSE across splits is a common way
to accidentally claim a scaffold split was harmless.

### Against published results

Scaffold-split RMSE from the [MoleculeNet scaffold leaderboard](https://www.sota2.com/research/sota/molecular-property-regression-on-moleculenet-scaffold):

| Model | Lipophilicity | ESOL |
|---|---|---|
| SMI-TED 289M (fine-tuned) | 0.552 | 0.611 |
| D-MPNN (chemprop) | 0.64 | 0.91 |
| MolCLR | 0.65 | 1.28 |
| ChemBERTa | 0.80 | 1.13 |
| GROVER | 0.823 | 1.423 |
| AttentiveFP | 1.11 | 1.74 |
| **this pipeline (counted ECFP + XGBoost ensemble)** | **0.749** | **0.903** |

Counted Morgan fingerprints plus physchem descriptors and a bagged XGBoost
ensemble land mid-table: level with D-MPNN on ESOL (0.903 vs 0.91), behind it
on Lipophilicity (0.749 vs 0.64), and ahead of several pretrained transformer
models on both. That is the evidence for the design claim in the main README —
at 10³–10⁴ compounds per endpoint, a well-built classical baseline is hard to
beat, and it trains in seconds.

**This comparison is indicative, not head-to-head**, for three specific
reasons: (1) the leaderboard's scaffold-split implementation and fold
averaging differ from this single deterministic Bemis-Murcko split;
(2) my ESOL set is **1011 compounds, not 1128** — the standardiser removed 106
molecules with fewer than 6 heavy atoms (dichloromethane, thiophene, thiourea,
propyl bromide), so this is a drug-like subset and a genuinely different task;
(3) no hyperparameter search was run. Anyone claiming a leaderboard position
needs matched splits.

That ESOL exclusion is worth dwelling on: 9.4% of a standard solubility
benchmark is below fragment size. A model trained on it is partly a model of
industrial-solvent solubility, which bears on how much it can be trusted for
lead optimisation — as Experiment 4 demonstrates concretely.

## Experiments 2 & 3 — retrospective virtual screen on BACE-1

A scaffold-disjoint 25% holdout (375 compounds) plays the part of an
unscreened library. The model is trained only on the remaining scaffolds, the
holdout is scored, and only then are the measured potencies consulted.

### Measured compounds only

| Metric | Value |
|---|---|
| Library size | 375 |
| RMSE / Spearman ρ | 0.700 / 0.766 |
| Active prevalence (pIC50 ≥ 7) | 0.536 |
| **EF ceiling** | **1.87** |
| EF @ 5% (pIC50 ≥ 7) | 1.77 |
| EF @ 5% (pIC50 ≥ 8) | 4.28 |
| Fraction in applicability domain | 1.00 |

Honest but uninformative on enrichment: everything in this set came from a
BACE-1 paper, so 54% of it is active and EF cannot exceed 1.87. The model
captures 1.77 of that 1.87 at the 5% cut — near-perfect ordering against a
metric with almost no room. The ≥8 threshold (10 nM) is more discriminating,
and 4.28 there is a real signal.

### Diluted with property-matched decoys

The same holdout mixed with 10× property-matched compounds from the
lipophilicity set that nobody has tested against BACE-1 — 4125 compounds, 5.1%
active, which resembles a real screening deck.

| Metric | pIC50 ≥ 7 | pIC50 ≥ 8 |
|---|---|---|
| Prevalence | 0.051 | 0.0147 |
| EF ceiling | 19.4 | 68.1 |
| **EF @ 1%** | **15.8** | **38.6** |
| EF @ 5% | 11.9 | 15.9 |
| EF @ 10% | 8.2 | 9.4 |
| Delivered hit list (100 cmpds, filters + AD + diversity cap) | 9.7 | 10.9 |

EF@1% of 15.8 against a ceiling of 19.4 means the top 1% of the ranking is 81%
of the way to a perfect screen. At the 10 nM threshold, EF@1% = 38.6.

The delivered hit list enriches less (9.7) than the raw top-1% (15.8), for the
same reason seen on simulated data: the diversity cap, applicability-domain
gate and 20% exploration budget deliberately trade top-end enrichment for
scaffold coverage. The pipeline reports both so the trade is visible rather
than hidden in a default.

**The assumption that bounds Experiment 3**: decoys are *presumed* inactive
because they were never tested against BACE-1, not because they were tested
and found inactive. Some are probably weak inhibitors, so these enrichments
carry an optimistic bias. This is the standard compromise in retrospective
screening benchmarks (it is what DUD-E does) and it is exactly why Experiment
2, which uses only measured values, is reported alongside.

## Experiment 4 — and the finding that matters most

Three models trained on three disjoint real datasets, combined into one MPO
score over the real BACE-1 hits. The optimiser worked: mean MPO 0.108 → 0.254,
best 0.228 → 0.401, with 316 analogs beating the best starting hit. With three
objectives the Pareto front is 7% of molecules (54 of 750) — properly
discriminating, unlike the 64% seen with six objectives on simulated data.

**But the applicability-domain check says most of that score is not real:**

| Objective | Fraction of candidates inside that model's domain |
|---|---|
| `pIC50` | **1.00** |
| `logD` | 0.45 |
| `logS` | **0.00** |

Not one of the final candidates falls inside the solubility model's
applicability domain, and fewer than half fall inside the lipophilicity
model's. The reason is obvious once stated: BACE-1 inhibitors are large
peptidomimetics (MW up to 750 here), while ESOL is substituted benzenes and
small solvents. The two datasets barely overlap in chemical space.

This is a real bug class, and the pipeline was initially silent about it.
Desirability scores a *missing* prediction as zero, but an out-of-domain
prediction is not missing — it is a confident-looking extrapolation that gets
scored like any other number. The combined MPO looked perfectly healthy while
one of its three terms was meaningless. `stages/lead_opt.py` now checks
per-objective domain coverage and raises an ERROR below 5% and a warning below
50%, and the figure is recorded in the run metrics and the report.

The general lesson is the one every multi-parameter pipeline has to confront:
**when your endpoints come from different datasets, the MPO is only as
meaningful as the chemical-space overlap between them.** A real programme
solves this by measuring its own ADMET data on its own series. Nothing in a
model can substitute for that, and the honest output of a pipeline in this
situation is a flag, not a number.

## What this does and does not establish

**Does:**
- The machinery runs end-to-end on real measured data, not just its own simulator.
- The QSAR models are competitive with published graph-neural-network baselines on two standard benchmarks.
- The uncertainty is approximately calibrated (0.85–0.92 coverage for nominal 0.90).
- Retrospective virtual screening on a real target with scaffold-disjoint holdout achieves EF@1% of 15.8 (≥100 nM) and 38.6 (≥10 nM).
- The curation, split and applicability-domain machinery each caught a real problem in real data: duplicate structures in published benchmark files, an RMSE comparison that would have misled, and an MPO objective evaluated entirely out of domain.

**Does not:**
- Prove prospective performance. Every result here is retrospective. The only honest test is making compounds.
- Cover clearance, permeability, hERG, protein binding or anything in vivo — hERG and Caco-2 sources (TDC, Zenodo, Mendeley) were unreachable from this environment, so the ADMET panel is two physicochemical endpoints, which are the easiest ones.
- Include structure-based scoring. BACE-1 has abundant crystallography and a real campaign would dock.
- Establish anything about a target without existing chemical matter. The activity model needs ~10³ measured compounds; with 20 you have a different problem.
- Beat a tuned graph network. With matched splits and a hyperparameter search, D-MPNN likely wins on Lipophilicity.

## Sources

- [MoleculeNet: a benchmark for molecular machine learning](https://pubs.rsc.org/sc/article/9/2/513/584204/MoleculeNet-a-benchmark-for-molecular-machine) — Wu et al., *Chem. Sci.* 2018
- [Computational modeling of β-secretase 1 (BACE-1) inhibitors](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00290) — Subramanian et al., *J. Chem. Inf. Model.* 2016
- [MoleculeNet scaffold-split regression leaderboard](https://www.sota2.com/research/sota/molecular-property-regression-on-moleculenet-scaffold)
- [Are Learned Molecular Representations Ready For Prime Time?](https://www.researchgate.net/publication/332169081_Are_Learned_Molecular_Representations_Ready_For_Prime_Time) — Yang et al. (chemprop / D-MPNN)
- [MoleculeNet benchmark notes](https://hunterheidenreich.com/notes/chemistry/molecular-design/property-prediction/moleculenet-benchmark-molecular-ml/)
- Data mirrors: [Molecules_Dataset_Collection](https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/master/latest), [DeepChem datasets](https://raw.githubusercontent.com/deepchem/deepchem/master/datasets)
