"""QSAR / property models with calibrated uncertainty.

Design choices that matter in production:

* **Ensembles, not single models.** The spread across members is the cheapest
  usable epistemic uncertainty signal, and it is what drives active-learning
  compound selection downstream.
* **Split conformal prediction.** Ensemble spread is not calibrated: an
  interval labelled 90% should contain the truth 90% of the time. Conformal
  calibration on a held-out set gives that guarantee with no distributional
  assumptions, which is what makes an interval safe to put in front of a
  project team.
* **Models carry their own metadata.** Training set size, date, split, metrics
  and feature config travel with the artifact, so a prediction can always be
  traced to the model that made it.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge

from ..config import ModelConfig

LOG = logging.getLogger("dd_pipeline.models.qsar")

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


@dataclass
class ModelCard:
    """Minimal model documentation. Fill it in; auditors ask."""

    endpoint: str
    task: str
    n_train: int = 0
    n_valid: int = 0
    n_test: int = 0
    split_strategy: str = "unknown"
    featurizer: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    assay_noise_estimate: float | None = None
    trained_at: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Prediction:
    mean: np.ndarray
    std: np.ndarray                      # ensemble spread (epistemic)
    lower: np.ndarray | None = None      # conformal interval
    upper: np.ndarray | None = None
    members: np.ndarray | None = None    # (n_members, n_samples)


def _make_member(cfg: ModelConfig, seed: int, member: int = 0):
    """Build one ensemble member.

    Members get jittered hyperparameters as well as a different bootstrap
    sample. Identical hyperparameters on bootstrapped data produce members
    that agree almost everywhere, which makes the ensemble spread useless as
    an uncertainty signal -- the failure mode is a model that is confidently
    wrong on novel chemistry, which is precisely where you need the warning.
    """
    if cfg.kind == "xgboost":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed; set activity_model.kind=random_forest")
        jitter = np.random.default_rng(seed)
        depth = int(np.clip(cfg.max_depth + jitter.integers(-2, 3), 3, 14))
        common = dict(
            n_estimators=cfg.n_estimators,
            max_depth=depth,
            learning_rate=float(cfg.learning_rate * jitter.uniform(0.7, 1.4)),
            subsample=float(np.clip(cfg.subsample * jitter.uniform(0.8, 1.15), 0.4, 1.0)),
            colsample_bytree=float(
                np.clip(cfg.colsample_bytree * jitter.uniform(0.6, 1.3), 0.2, 1.0)
            ),
            min_child_weight=float(cfg.min_child_weight * jitter.uniform(0.6, 1.8)),
            reg_lambda=float(cfg.reg_lambda * jitter.uniform(0.5, 3.0)),
            random_state=seed,
            n_jobs=cfg.n_jobs,
            tree_method="hist",
        )
        if cfg.task == "regression":
            return xgb.XGBRegressor(objective="reg:squarederror", **common)
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)

    if cfg.kind == "random_forest":
        jitter = np.random.default_rng(seed)
        common = dict(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth or None,
            random_state=seed, n_jobs=cfg.n_jobs,
            min_samples_leaf=int(jitter.integers(1, 4)),
            max_features=float(jitter.uniform(0.1, 0.6)),
        )
        return (RandomForestRegressor(**common) if cfg.task == "regression"
                else RandomForestClassifier(class_weight="balanced", **common))

    if cfg.kind == "ridge":
        return (Ridge(alpha=1.0, random_state=seed) if cfg.task == "regression"
                else LogisticRegression(max_iter=2000, class_weight="balanced"))

    raise ValueError(f"unknown model kind {cfg.kind}")


class EnsembleQSAR:
    """Bagged ensemble with conformal calibration."""

    def __init__(self, config: ModelConfig | None = None, endpoint: str = "activity") -> None:
        self.cfg = config or ModelConfig()
        self.endpoint = endpoint
        self.members: list[Any] = []
        self.card = ModelCard(endpoint=endpoint, task=self.cfg.task)
        self._conformal_q: dict[float, float] = {}
        self._std_scale: float = 1e-3   # stabiliser for normalised conformal scores
        self._prob_calibrator: IsotonicRegression | None = None
        self._feature_names: list[str] | None = None

    # ---- training ---------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "EnsembleQSAR":
        import time

        self._feature_names = feature_names
        rng = np.random.default_rng(self.cfg.seed)
        n = len(X)
        self.members = []

        for m in range(self.cfg.n_ensemble):
            seed = int(self.cfg.seed + 1000 * m)
            model = _make_member(self.cfg, seed, member=m)
            # Bootstrap resampling gives the members something to disagree
            # about beyond the RNG seed, which is what makes the spread useful.
            idx = (rng.choice(n, size=n, replace=True) if self.cfg.n_ensemble > 1
                   else np.arange(n))
            Xb, yb = X[idx], y[idx]
            wb = sample_weight[idx] if sample_weight is not None else None

            fit_kw: dict[str, Any] = {}
            if wb is not None:
                fit_kw["sample_weight"] = wb
            if (self.cfg.kind == "xgboost" and X_valid is not None
                    and self.cfg.early_stopping_rounds):
                model.set_params(early_stopping_rounds=self.cfg.early_stopping_rounds)
                fit_kw["eval_set"] = [(X_valid, y_valid)]
                fit_kw["verbose"] = False
            model.fit(Xb, yb, **fit_kw)
            self.members.append(model)

        self.card.n_train = int(n)
        self.card.n_valid = int(len(X_valid)) if X_valid is not None else 0
        self.card.trained_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Calibrate on the validation fold (never on test).
        if X_valid is not None and y_valid is not None and len(X_valid) >= 20:
            if self.cfg.task == "regression":
                self._fit_conformal(X_valid, y_valid)
            else:
                self._fit_probability_calibration(X_valid, y_valid)

        LOG.info("fit %s: %d members on %d compounds", self.endpoint, len(self.members), n)
        return self

    def _fit_conformal(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Split-conformal residual quantiles, normalised by ensemble spread.

        Using |y - mean| / (std + s0) as the nonconformity score makes the
        interval width adapt per compound: wide where the ensemble disagrees.

        `s0` is the median calibration spread rather than a tiny epsilon. With
        a bare epsilon, compounds where the members happen to agree exactly
        drive the quantile to absurd values, and the resulting intervals are
        both enormous and meaningless. Anchoring on the median keeps the
        quantile O(1) and makes the interval degrade to a constant width when
        the spread carries no information -- which is the honest behaviour.
        """
        pred = self.predict(X_cal, conformal=False)
        self._std_scale = float(max(np.median(pred.std), 1e-3))
        scores = np.abs(y_cal - pred.mean) / (pred.std + self._std_scale)
        for alpha in (0.1, 0.2, 0.5):
            level = 1 - alpha
            k = int(np.ceil((len(scores) + 1) * level))
            k = min(max(k, 1), len(scores))
            self._conformal_q[level] = float(np.sort(scores)[k - 1])
        LOG.info("conformal quantiles for %s (std scale %.3f): %s", self.endpoint,
                 self._std_scale, {k: round(v, 3) for k, v in self._conformal_q.items()})

    def _fit_probability_calibration(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        raw = self.predict(X_cal, conformal=False).mean
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw, y_cal)
        self._prob_calibrator = iso

    # ---- prediction -------------------------------------------------------
    def predict(self, X: np.ndarray, conformal: bool = True, level: float = 0.9) -> Prediction:
        if not self.members:
            raise RuntimeError("model is not fitted")
        preds = []
        for m in self.members:
            if self.cfg.task == "classification":
                preds.append(m.predict_proba(X)[:, 1])
            else:
                preds.append(m.predict(X))
        P = np.vstack(preds)
        mean = P.mean(axis=0)
        std = P.std(axis=0) if len(self.members) > 1 else np.zeros_like(mean)

        if self.cfg.task == "classification" and self._prob_calibrator is not None:
            mean = self._prob_calibrator.predict(mean)

        lower = upper = None
        if conformal and self.cfg.task == "regression" and self._conformal_q:
            q = self._conformal_q.get(level, max(self._conformal_q.values()))
            half = q * (std + self._std_scale)
            lower, upper = mean - half, mean + half
        return Prediction(mean=mean, std=std, lower=lower, upper=upper, members=P)

    # ---- interpretation ---------------------------------------------------
    def feature_importance(self, top_n: int = 25) -> list[tuple[str, float]]:
        imps = []
        for m in self.members:
            if hasattr(m, "feature_importances_"):
                imps.append(np.asarray(m.feature_importances_, dtype=float))
        if not imps:
            return []
        avg = np.mean(imps, axis=0)
        names = self._feature_names or [f"f_{i}" for i in range(len(avg))]
        order = np.argsort(avg)[::-1][:top_n]
        return [(names[i], float(avg[i])) for i in order]

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "config": self.cfg.model_dump(),
                    "endpoint": self.endpoint,
                    "members": self.members,
                    "conformal_q": self._conformal_q,
                    "std_scale": self._std_scale,
                    "prob_calibrator": self._prob_calibrator,
                    "feature_names": self._feature_names,
                    "card": self.card.to_dict(),
                },
                fh,
            )
        card_path = path.with_suffix(".card.json")
        card_path.write_text(json.dumps(self.card.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleQSAR":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        obj = cls(ModelConfig(**blob["config"]), endpoint=blob["endpoint"])
        obj.members = blob["members"]
        obj._conformal_q = blob["conformal_q"]
        obj._std_scale = blob.get("std_scale", 1e-3)
        obj._prob_calibrator = blob["prob_calibrator"]
        obj._feature_names = blob["feature_names"]
        obj.card = ModelCard(**blob["card"])
        return obj
