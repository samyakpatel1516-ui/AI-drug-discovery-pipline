"""Applicability domain.

A model will return a confident-looking number for any molecule you hand it,
including chemistry that has nothing to do with its training set. The domain
check is what separates "predicted 8.2" from "predicted 8.2, and we have seen
compounds like this before".

Two orthogonal signals are combined:
  * chemical similarity to the training set (Tanimoto to k nearest neighbours)
  * descriptor-space distance (Mahalanobis-style, on the physchem block)
"""
from __future__ import annotations

import logging

import numpy as np
from rdkit import DataStructs

from ..chem.featurize import Featurizer
from ..config import ApplicabilityConfig

LOG = logging.getLogger("dd_pipeline.models.applicability")


class ApplicabilityDomain:
    def __init__(self, config: ApplicabilityConfig | None = None,
                 featurizer: Featurizer | None = None) -> None:
        self.cfg = config or ApplicabilityConfig()
        self.featurizer = featurizer or Featurizer()
        self._train_fps: list = []
        self._desc_mean: np.ndarray | None = None
        self._desc_scale: np.ndarray | None = None
        self._desc_cutoff: float | None = None

    def fit(self, smiles: list[str], X_desc: np.ndarray | None = None) -> "ApplicabilityDomain":
        self._train_fps = [f for f in self.featurizer.bit_fingerprints(smiles) if f is not None]
        if X_desc is not None and X_desc.size:
            self._desc_mean = X_desc.mean(axis=0)
            self._desc_scale = X_desc.std(axis=0) + 1e-8
            z = np.abs((X_desc - self._desc_mean) / self._desc_scale)
            d = z.max(axis=1)
            # 99th percentile of training leverage as the outlier boundary.
            self._desc_cutoff = float(np.percentile(d, 99))
        LOG.info("applicability domain fit on %d training molecules", len(self._train_fps))
        return self

    def similarity_scores(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean similarity to k nearest neighbours, max similarity)."""
        k = self.cfg.k_neighbours
        fps = self.featurizer.bit_fingerprints(smiles)
        mean_sim = np.zeros(len(smiles))
        max_sim = np.zeros(len(smiles))
        if not self._train_fps:
            return mean_sim, max_sim
        for i, fp in enumerate(fps):
            if fp is None:
                continue
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self._train_fps))
            if sims.size == 0:
                continue
            top = np.sort(sims)[::-1][:k]
            mean_sim[i] = float(top.mean())
            max_sim[i] = float(top[0])
        return mean_sim, max_sim

    def descriptor_outlier(self, X_desc: np.ndarray) -> np.ndarray:
        if self._desc_mean is None or self._desc_cutoff is None:
            return np.zeros(len(X_desc), dtype=bool)
        z = np.abs((X_desc - self._desc_mean) / self._desc_scale)
        return z.max(axis=1) > self._desc_cutoff

    def assess(self, smiles: list[str], X_desc: np.ndarray | None = None) -> dict[str, np.ndarray]:
        mean_sim, max_sim = self.similarity_scores(smiles)
        in_domain = mean_sim >= self.cfg.min_mean_similarity
        desc_out = (
            self.descriptor_outlier(X_desc) if X_desc is not None
            else np.zeros(len(smiles), dtype=bool)
        )
        in_domain = in_domain & ~desc_out
        # 0..1 confidence; saturates at twice the similarity threshold.
        conf = np.clip(mean_sim / max(self.cfg.min_mean_similarity * 2, 1e-6), 0.0, 1.0)
        conf[desc_out] *= 0.5
        return {
            "ad_mean_similarity": mean_sim,
            "ad_max_similarity": max_sim,
            "ad_descriptor_outlier": desc_out,
            "in_domain": in_domain,
            "ad_confidence": conf,
        }
