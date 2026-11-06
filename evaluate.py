"""Model evaluation with metrics that reflect how the model gets used.

R^2 on a random split is the metric most often reported and the least often
relevant. A virtual screen is a ranking problem with a tiny selection budget,
so what matters is enrichment in the top fraction you can actually buy, plus
whether the uncertainty is honest.

Every evaluation is reported against two baselines -- a constant predictor and
1-nearest-neighbour on fingerprints. A model that cannot beat 1-NN has learned
nothing beyond "similar molecules have similar activity", and does not justify
its maintenance cost.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .qsar import Prediction

LOG = logging.getLogger("dd_pipeline.models.evaluate")


# ---- ranking metrics -----------------------------------------------------
def enrichment_factor(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.01,
                      threshold: float | None = None, warn: bool = True) -> float:
    """How many more actives the model finds in the top `frac` than random.

    EF=1 is no better than picking at random. For a 1% screen of a vendor
    library, EF of 10-30 is a useful model; EF of 2 is not worth the plates.

    EF is capped at 1/prevalence, so it is only informative when actives are
    rare. On an active-enriched test set (which is what you get if the
    historical programme measured what chemists liked, and a scaffold split
    then concentrates it) EF saturates near its ceiling and says nothing. The
    warning below fires in that case; trust Spearman and the ROC/PR numbers
    instead, and compute EF against the full unlabelled library if you want
    the screening-economics number.
    """
    y_true = np.asarray(y_true)
    labels = (y_true >= threshold) if threshold is not None else y_true.astype(bool)
    n = len(y_score)
    k = max(1, int(np.ceil(frac * n)))
    top = np.argsort(y_score)[::-1][:k]
    hit_rate_top = labels[top].mean()
    hit_rate_all = labels.mean()
    if hit_rate_all <= 0:
        return float("nan")
    if warn and hit_rate_all > 0.5:
        LOG.warning(
            "enrichment factor is uninformative here: %.0f%% of the evaluation set is "
            "active, so EF is capped at %.2f. Use spearman / pr_auc instead.",
            100 * hit_rate_all, 1.0 / hit_rate_all,
        )
    return float(hit_rate_top / hit_rate_all)


def bedroc(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0,
           threshold: float | None = None) -> float:
    """Boltzmann-enhanced discrimination of ROC: ROC-AUC's early-recognition
    cousin. alpha=20 weights roughly the top 8% of the ranking."""
    labels = ((np.asarray(y_true) >= threshold) if threshold is not None
              else np.asarray(y_true).astype(bool)).astype(int)
    n = len(labels)
    n_act = int(labels.sum())
    if n_act == 0 or n_act == n:
        return float("nan")
    order = np.argsort(y_score)[::-1]
    ranks = np.where(labels[order] == 1)[0] + 1
    ra = n_act / n
    s = np.sum(np.exp(-alpha * ranks / n))
    random_sum = ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    factor = (ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra)))
    return float(s / random_sum * factor + 1 / (1 - np.exp(alpha * (1 - ra))))


# ---- uncertainty quality -------------------------------------------------
def conformal_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of truths inside the interval. Should match the nominal level;
    much lower means the intervals are lying."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def uncertainty_error_correlation(y_true: np.ndarray, pred: Prediction) -> float:
    """Spearman correlation between predicted uncertainty and actual error.

    Positive and meaningful (>~0.3) means the model knows what it does not
    know, which is the prerequisite for using it to choose the next assay plate.
    """
    err = np.abs(y_true - pred.mean)
    if np.allclose(pred.std, pred.std[0]):
        return float("nan")
    return float(stats.spearmanr(pred.std, err).statistic)


def error_at_confidence(y_true: np.ndarray, pred: Prediction,
                        quantiles=(0.25, 0.5, 0.75, 1.0)) -> dict[str, float]:
    """RMSE restricted to the most-confident X% of predictions.

    A good model's error should fall steeply as you keep only confident calls;
    a flat curve means the uncertainty is decorative.
    """
    err2 = (y_true - pred.mean) ** 2
    order = np.argsort(pred.std)
    out: dict[str, float] = {}
    for q in quantiles:
        k = max(1, int(q * len(order)))
        out[f"rmse_top{int(q * 100)}pct_confident"] = float(np.sqrt(err2[order[:k]].mean()))
    return out


# ---- baselines -----------------------------------------------------------
def baseline_metrics(y_train: np.ndarray, y_test: np.ndarray,
                     fps_train: list | None = None, fps_test: list | None = None,
                     task: str = "regression") -> dict[str, float]:
    out: dict[str, float] = {}
    if task == "regression":
        const = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
        out["baseline_constant_rmse"] = float(np.sqrt(mean_squared_error(y_test, const)))
    else:
        out["baseline_prevalence"] = float(np.mean(y_train))

    if fps_train is not None and fps_test is not None:
        from rdkit import DataStructs

        valid_tr = [(f, y) for f, y in zip(fps_train, y_train) if f is not None]
        if valid_tr:
            tr_fps = [f for f, _ in valid_tr]
            tr_y = np.array([y for _, y in valid_tr], dtype=float)
            preds = np.full(len(fps_test), float(tr_y.mean()))
            for i, f in enumerate(fps_test):
                if f is None:
                    continue
                sims = np.asarray(DataStructs.BulkTanimotoSimilarity(f, tr_fps))
                preds[i] = tr_y[int(np.argmax(sims))]
            if task == "regression":
                out["baseline_1nn_rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
                out["baseline_1nn_spearman"] = float(stats.spearmanr(y_test, preds).statistic)
            else:
                try:
                    out["baseline_1nn_auc"] = float(roc_auc_score(y_test, preds))
                except ValueError:
                    pass
    return out


# ---- top-level -----------------------------------------------------------
def evaluate_regression(
    y_true: np.ndarray, pred: Prediction, active_threshold: float | None = None,
    assay_noise: float | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    m: dict[str, float] = {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred.mean))),
        "mae": float(mean_absolute_error(y_true, pred.mean)),
        "r2": float(r2_score(y_true, pred.mean)),
        "pearson": float(stats.pearsonr(y_true, pred.mean).statistic),
        "spearman": float(stats.spearmanr(y_true, pred.mean).statistic),
        "kendall_tau": float(stats.kendalltau(y_true, pred.mean).statistic),
    }
    # Fraction predicted within half a log unit -- the tolerance a chemist
    # actually cares about for a potency prediction.
    m["frac_within_0.5_log"] = float(np.mean(np.abs(y_true - pred.mean) <= 0.5))
    m["frac_within_1.0_log"] = float(np.mean(np.abs(y_true - pred.mean) <= 1.0))

    if active_threshold is not None:
        labels_bool = y_true >= active_threshold
        m["active_prevalence"] = float(labels_bool.mean())
        # Record the ceiling alongside the value so nobody reads EF=1.2 as a
        # weak model when it is in fact a saturated metric.
        m["ef_ceiling"] = (float(1.0 / m["active_prevalence"])
                           if m["active_prevalence"] > 0 else float("nan"))
        for i, frac in enumerate((0.01, 0.05, 0.10)):
            m[f"ef_{int(frac * 100)}pct"] = enrichment_factor(
                y_true, pred.mean, frac, threshold=active_threshold, warn=(i == 0)
            )
        m["bedroc_a20"] = bedroc(y_true, pred.mean, 20.0, threshold=active_threshold)
        labels = labels_bool.astype(int)
        if 0 < labels.sum() < len(labels):
            m["roc_auc_at_threshold"] = float(roc_auc_score(labels, pred.mean))
            m["pr_auc_at_threshold"] = float(average_precision_score(labels, pred.mean))

    if pred.lower is not None and pred.upper is not None:
        m["conformal_coverage_90"] = conformal_coverage(y_true, pred.lower, pred.upper)
        m["mean_interval_width"] = float(np.mean(pred.upper - pred.lower))
    m["unc_error_spearman"] = uncertainty_error_correlation(y_true, pred)
    m.update(error_at_confidence(y_true, pred))

    if assay_noise:
        # Error relative to the experiment's own reproducibility. Below ~1.0
        # the model is at the noise floor and more data will not help much.
        m["rmse_over_assay_noise"] = float(m["rmse"] / assay_noise)
    return m


def evaluate_classification(y_true: np.ndarray, pred: Prediction) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    m: dict[str, float] = {"n": int(len(y_true)), "prevalence": float(y_true.mean())}
    if 0 < y_true.sum() < len(y_true):
        m["roc_auc"] = float(roc_auc_score(y_true, pred.mean))
        m["pr_auc"] = float(average_precision_score(y_true, pred.mean))
        for frac in (0.01, 0.05, 0.10):
            m[f"ef_{int(frac * 100)}pct"] = enrichment_factor(y_true, pred.mean, frac)
        m["bedroc_a20"] = bedroc(y_true, pred.mean, 20.0)
        # Expected calibration error over 10 equal-width bins.
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(pred.mean, bins) - 1, 0, 9)
        ece = 0.0
        for b in range(10):
            sel = idx == b
            if sel.sum():
                ece += sel.mean() * abs(pred.mean[sel].mean() - y_true[sel].mean())
        m["expected_calibration_error"] = float(ece)
    return m
