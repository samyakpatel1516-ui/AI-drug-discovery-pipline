"""Multi-parameter optimisation (MPO) scoring.

Potency is one of six or seven things that have to be true at once. Optimising
it alone is the classic failure mode: you get a 50 pM compound that is
insoluble, cleared in minutes and hits hERG.

Each endpoint is mapped to a 0-1 desirability through a smooth transform, then
combined as a *weighted geometric mean*. Geometric, not arithmetic: a zero on
any term should drag the whole score down, because a candidate that fails one
criterion badly cannot be rescued by excelling at the others.

Critical objectives are harder still -- a zero there zeroes the total.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import LeadOptConfig, ObjectiveSpec

LOG = logging.getLogger("dd_pipeline.scoring.desirability")


def _sigmoid_ramp(x: np.ndarray, low: float, high: float, rising: bool) -> np.ndarray:
    """Smooth 0->1 ramp between low and high.

    Smooth rather than linear-clipped because a hard edge makes gradient-free
    optimisers (and chemists) chase irrelevant decimal places near the cutoff.
    """
    span = max(high - low, 1e-9)
    z = (x - low) / span
    # Logistic centred on the midpoint, ~0.02/0.98 at the bounds.
    out = 1.0 / (1.0 + np.exp(-8.0 * (z - 0.5)))
    return out if rising else 1.0 - out


def desirability(values: np.ndarray, spec: ObjectiveSpec) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v)
    finite = np.isfinite(v)

    if spec.kind == "higher_better":
        out[finite] = _sigmoid_ramp(v[finite], spec.low, spec.high, rising=True)
    elif spec.kind == "lower_better":
        out[finite] = _sigmoid_ramp(v[finite], spec.low, spec.high, rising=False)
    elif spec.kind == "window":
        # Plateau inside [low, high], falling off outside by a quarter-width.
        margin = max((spec.high - spec.low) * 0.25, 1e-9)
        rise = _sigmoid_ramp(v[finite], spec.low - margin, spec.low, rising=True)
        fall = _sigmoid_ramp(v[finite], spec.high, spec.high + margin, rising=False)
        out[finite] = np.minimum(rise, fall)
    else:
        raise ValueError(f"unknown desirability kind {spec.kind}")

    # Missing predictions score 0, not a guess. Absence of data is not a pass.
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


@dataclass
class MPOResult:
    score: np.ndarray
    per_objective: dict[str, np.ndarray]
    failed_critical: np.ndarray


class MPOScorer:
    def __init__(self, config: LeadOptConfig | None = None,
                 critical_floor: float = 0.02) -> None:
        """`critical_floor`: desirability below this on a `critical` objective
        disqualifies the compound outright.

        Note how this interacts with the sigmoid: desirability at the `low`
        bound is ~0.018, so a floor of 0.02 makes `low` behave as a hard cutoff
        for critical objectives and a soft preference for the rest. That is
        usually what people mean by "critical", but set the floor lower if you
        want the gate to bite only well below the stated bound.
        """
        self.cfg = config or LeadOptConfig()
        self.critical_floor = critical_floor

    @property
    def endpoints(self) -> list[str]:
        return [o.endpoint for o in self.cfg.objectives]

    def score_frame(self, df: pd.DataFrame) -> MPOResult:
        n = len(df)
        per: dict[str, np.ndarray] = {}
        log_sum = np.zeros(n)
        weight_sum = 0.0
        failed = np.zeros(n, dtype=bool)
        missing: list[str] = []

        for obj in self.cfg.objectives:
            if obj.endpoint not in df.columns:
                missing.append(obj.endpoint)
                continue
            d = desirability(df[obj.endpoint].to_numpy(), obj)
            per[obj.endpoint] = d
            # eps keeps log finite; a zero term still collapses the product.
            log_sum += obj.weight * np.log(np.clip(d, 1e-6, 1.0))
            weight_sum += obj.weight
            if obj.critical:
                failed |= d < self.critical_floor

        if missing:
            LOG.warning("MPO: endpoints absent from table, skipped: %s", missing)
        if weight_sum == 0:
            raise ValueError("no scorable objectives present in the table")

        score = np.exp(log_sum / weight_sum)
        score[failed] = 0.0
        return MPOResult(score=score, per_objective=per, failed_critical=failed)

    def annotate(self, df: pd.DataFrame, prefix: str = "d_") -> pd.DataFrame:
        res = self.score_frame(df)
        out = df.copy()
        for name, vals in res.per_objective.items():
            out[f"{prefix}{name}"] = vals
        out["mpo_score"] = res.score
        out["fails_critical"] = res.failed_critical
        return out


# ---- published reference scores ------------------------------------------
def cns_mpo(df: pd.DataFrame) -> np.ndarray:
    """Pfizer CNS MPO (Wager et al.). Six physchem terms, 0-6.

    Included because it is a useful sanity reference: if your bespoke MPO
    ranks compounds very differently from a published score on the same
    physchem axes, find out why before trusting it.
    """
    def ramp(x, hi_good, lo_bad):
        x = np.asarray(x, dtype=float)
        if hi_good > lo_bad:
            return np.clip((x - lo_bad) / (hi_good - lo_bad), 0, 1)
        return np.clip((lo_bad - x) / (lo_bad - hi_good), 0, 1)

    terms = []
    if "clogp" in df:
        terms.append(ramp(df["clogp"], 3.0, 5.0))
    if "logD" in df:
        terms.append(ramp(df["logD"], 2.0, 4.0))
    if "mw" in df:
        terms.append(ramp(df["mw"], 360.0, 500.0))
    if "tpsa" in df:
        terms.append(np.minimum(ramp(df["tpsa"], 40.0, 20.0), ramp(df["tpsa"], 90.0, 120.0)))
    if "hbd" in df:
        terms.append(ramp(df["hbd"], 0.5, 3.5))
    if "pka_basic" in df:
        terms.append(ramp(df["pka_basic"], 8.0, 10.0))
    return np.sum(terms, axis=0) if terms else np.zeros(len(df))


def pessimistic_frame(
    df: pd.DataFrame,
    objectives: list[ObjectiveSpec],
    beta: float = 1.0,
    std_suffix: str = "_std",
) -> pd.DataFrame:
    """Shift every predicted endpoint against its own preferred direction by
    `beta` standard deviations.

    This is the single most important guard against generative
    over-optimisation. An optimiser scored on predicted means will happily
    march into the region where the model is most wrong, because that is where
    the predictions look best -- the diagnostic signature is a *negative*
    correlation between predicted and true score on the final shortlist, even
    while the mean genuinely improves.

    Scoring a pessimistic estimate instead means a molecule can only win by
    being good somewhere the model is also confident. beta=0 recovers the
    optimistic behaviour; beta=1 is a reasonable default; higher values keep
    the search closer to known chemistry at the cost of exploration.

    For a `window` objective there is no single bad direction, so the interval
    is widened symmetrically: the penalty is applied by moving the value away
    from the window centre.
    """
    out = df.copy()
    if beta <= 0:
        return out
    for obj in objectives:
        col, scol = obj.endpoint, f"{obj.endpoint}{std_suffix}"
        if col not in out.columns or scol not in out.columns:
            continue
        v = out[col].to_numpy(dtype=float)
        s = np.nan_to_num(out[scol].to_numpy(dtype=float), nan=0.0)
        if obj.kind == "higher_better":
            out[col] = v - beta * s
        elif obj.kind == "lower_better":
            out[col] = v + beta * s
        else:  # window: push away from the centre of the acceptable range
            centre = 0.5 * (obj.low + obj.high)
            out[col] = v + np.sign(v - centre) * beta * s
    return out


def ligand_efficiency(pactivity: np.ndarray, heavy_atoms: np.ndarray) -> np.ndarray:
    """LE = 1.37 * pActivity / heavy atom count (kcal/mol per heavy atom).

    Guards against the commonest kind of false progress in lead optimisation:
    potency bought entirely with molecular weight.
    """
    ha = np.asarray(heavy_atoms, dtype=float)
    return np.where(ha > 0, 1.37 * np.asarray(pactivity, dtype=float) / ha, np.nan)


def lipophilic_efficiency(pactivity: np.ndarray, clogp: np.ndarray) -> np.ndarray:
    """LipE = pActivity - clogP. Potency that is not just grease."""
    return np.asarray(pactivity, dtype=float) - np.asarray(clogp, dtype=float)
