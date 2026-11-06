"""Typed configuration for the discovery pipeline.

Everything that changes between programs lives here, not in code. A run is
reproducible from (config file + code commit + input data hashes).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class SplitConfig(BaseModel):
    """How data is partitioned. Scaffold splitting is the default because
    random splits inflate QSAR metrics badly on congeneric series."""

    strategy: Literal["scaffold", "random", "temporal", "cluster"] = "scaffold"
    test_frac: float = 0.2
    valid_frac: float = 0.1
    seed: int = 1337
    # For temporal splits: name of the date column in the assay table.
    date_column: str | None = None


class FeaturizerConfig(BaseModel):
    kind: Literal["ecfp_desc", "ecfp", "descriptors"] = "ecfp_desc"
    fp_radius: int = 2
    fp_bits: int = 2048
    use_counts: bool = True
    use_chirality: bool = True
    # Physchem descriptor block appended to the fingerprint.
    descriptor_set: Literal["core", "full"] = "core"


class ModelConfig(BaseModel):
    kind: Literal["xgboost", "random_forest", "ridge"] = "xgboost"
    task: Literal["regression", "classification"] = "regression"
    n_ensemble: int = Field(5, ge=1, description="Members for epistemic uncertainty.")
    n_estimators: int = 600
    max_depth: int = 7
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.6
    min_child_weight: float = 3.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int | None = 50
    seed: int = 1337
    n_jobs: int = -1


class ApplicabilityConfig(BaseModel):
    """Applicability domain. Predictions outside the domain are reported but
    flagged; they should not drive purchase or synthesis decisions."""

    enabled: bool = True
    k_neighbours: int = 5
    # Mean Tanimoto similarity to k nearest training actives below which a
    # prediction is considered out-of-domain.
    min_mean_similarity: float = 0.25


class TargetIDConfig(BaseModel):
    """Target identification / prioritisation."""

    # Differential expression thresholds.
    min_abs_log2fc: float = 1.0
    max_fdr: float = 0.05
    # Dependency (e.g. CRISPR knockout) score below which a gene is called
    # essential in the disease models. DepMap convention: more negative =
    # stronger dependency, -1.0 ~ median of known common essentials.
    dependency_threshold: float = -0.5
    # Penalty applied when a gene is also essential in normal/control lines.
    normal_essentiality_penalty: float = 2.0
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "differential_expression": 0.20,
            "dependency": 0.30,
            "druggability": 0.25,
            "safety": 0.15,
            "tractability_evidence": 0.10,
        }
    )

    @field_validator("weights")
    @classmethod
    def _weights_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"target weights must sum to 1.0, got {total:.4f}")
        return v


class HitDiscoveryConfig(BaseModel):
    # Structural alert filters applied before scoring.
    apply_pains: bool = True
    apply_brenk: bool = True
    apply_nih: bool = False
    # Hard property gates (lead-like, deliberately tighter than Ro5).
    mw_range: tuple[float, float] = (200.0, 480.0)
    clogp_range: tuple[float, float] = (-1.0, 4.5)
    max_rotatable_bonds: int = 8
    max_hbd: int = 4
    max_tpsa: float = 120.0
    # Screening economics.
    n_hits: int = 200
    # Minimum predicted activity (pIC50 units) to be considered a hit.
    min_predicted_activity: float = 6.0
    # Diversity picking: max Tanimoto between any two selected hits.
    max_pairwise_similarity: float = 0.70
    # Fraction of the hit budget spent on exploration (high-uncertainty,
    # in-domain compounds) rather than pure exploitation.
    exploration_frac: float = 0.25


class ObjectiveSpec(BaseModel):
    """One term in the multi-parameter optimisation (MPO) score.

    `kind` picks the desirability shape; bounds are in the endpoint's own
    units. `weight` is relative importance; `critical` makes the objective a
    hard gate (a zero here zeroes the whole MPO score).
    """

    endpoint: str
    kind: Literal["higher_better", "lower_better", "window"] = "higher_better"
    low: float
    high: float
    weight: float = 1.0
    critical: bool = False


class LeadOptConfig(BaseModel):
    objectives: list[ObjectiveSpec] = Field(
        default_factory=lambda: [
            ObjectiveSpec(endpoint="pIC50", kind="higher_better", low=6.0, high=9.0,
                          weight=3.0, critical=True),
            ObjectiveSpec(endpoint="logS", kind="higher_better", low=-6.0, high=-3.5,
                          weight=1.5),
            ObjectiveSpec(endpoint="logD", kind="window", low=1.0, high=3.5, weight=1.0),
            ObjectiveSpec(endpoint="hlm_clint", kind="lower_better", low=10.0, high=60.0,
                          weight=2.0),
            ObjectiveSpec(endpoint="herg_pic50", kind="lower_better", low=4.5, high=6.0,
                          weight=2.0, critical=True),
            ObjectiveSpec(endpoint="caco2_papp", kind="higher_better", low=-6.5, high=-5.0,
                          weight=1.0),
        ]
    )
    # Analog enumeration around each starting point.
    n_analogs_per_seed: int = 300
    n_seeds: int = 20
    max_generations: int = 3
    # Keep only analogs at least this similar to their seed (series integrity).
    min_similarity_to_seed: float = 0.40
    # Synthetic accessibility gate (RDKit SA score, 1=easy .. 10=hard).
    max_sa_score: float = 4.5
    n_final_candidates: int = 25
    # Standard deviations of predicted uncertainty to subtract from each
    # endpoint before scoring, so the optimiser cannot profit from regions
    # where the model is unsure. 0 disables it (and reproduces the classic
    # over-optimisation failure). See scoring.desirability.pessimistic_frame.
    uncertainty_penalty: float = 1.0


class PipelineConfig(BaseModel):
    project: str = "demo-program"
    run_root: Path = Path("runs")
    seed: int = 1337

    target_id: TargetIDConfig = Field(default_factory=TargetIDConfig)
    featurizer: FeaturizerConfig = Field(default_factory=FeaturizerConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    activity_model: ModelConfig = Field(default_factory=ModelConfig)
    admet_model: ModelConfig = Field(default_factory=ModelConfig)
    applicability: ApplicabilityConfig = Field(default_factory=ApplicabilityConfig)
    hit_discovery: HitDiscoveryConfig = Field(default_factory=HitDiscoveryConfig)
    lead_opt: LeadOptConfig = Field(default_factory=LeadOptConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)
