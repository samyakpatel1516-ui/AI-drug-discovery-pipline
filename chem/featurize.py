"""Molecular featurisation.

Counted Morgan fingerprints concatenated with interpretable physchem
descriptors. This is an unglamorous baseline that is very hard to beat with a
graph network on the data sizes typical of a real project (10^2-10^4 measured
compounds per endpoint), and it trains in seconds, which matters when you
refit after every assay batch.

Swap in a learned representation behind the same `Featurizer` interface if you
have the data to justify it -- nothing downstream needs to change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

from ..config import FeaturizerConfig

LOG = logging.getLogger("dd_pipeline.chem.featurize")

# Descriptors chosen to be fast, well-behaved and physically meaningful.
# Keeping them named means feature importances are readable by a chemist.
CORE_DESCRIPTORS: dict[str, callable] = {
    "mw": Descriptors.MolWt,
    "clogp": Crippen.MolLogP,
    "tpsa": rdMolDescriptors.CalcTPSA,
    "hbd": rdMolDescriptors.CalcNumHBD,
    "hba": rdMolDescriptors.CalcNumHBA,
    "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings,
    "heavy_atoms": lambda m: float(m.GetNumHeavyAtoms()),
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "ring_count": rdMolDescriptors.CalcNumRings,
    "stereocentres": lambda m: float(rdMolDescriptors.CalcNumAtomStereoCenters(m)),
    "formal_charge": lambda m: float(Chem.GetFormalCharge(m)),
    "qed": Descriptors.qed,
    "heteroatom_frac": lambda m: (
        float(rdMolDescriptors.CalcNumHeteroatoms(m)) / max(m.GetNumHeavyAtoms(), 1)
    ),
}

FULL_EXTRA: dict[str, callable] = {
    "balaban_j": Descriptors.BalabanJ,
    "bertz_ct": Descriptors.BertzCT,
    "chi0v": rdMolDescriptors.CalcChi0v,
    "chi1v": rdMolDescriptors.CalcChi1v,
    "kappa2": rdMolDescriptors.CalcKappa2,
    "labute_asa": rdMolDescriptors.CalcLabuteASA,
    "molmr": Crippen.MolMR,
    "max_partial_charge": Descriptors.MaxPartialCharge,
    "min_partial_charge": Descriptors.MinPartialCharge,
    "num_amide_bonds": rdMolDescriptors.CalcNumAmideBonds,
    "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms,
    "bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms,
}


@dataclass
class FeatureMatrix:
    X: np.ndarray
    feature_names: list[str]
    valid_mask: np.ndarray  # True where the input SMILES was featurizable

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


class Featurizer:
    def __init__(self, config: FeaturizerConfig | None = None) -> None:
        self.cfg = config or FeaturizerConfig()
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.cfg.fp_radius,
            fpSize=self.cfg.fp_bits,
            includeChirality=self.cfg.use_chirality,
        )
        self._desc = dict(CORE_DESCRIPTORS)
        if self.cfg.descriptor_set == "full":
            self._desc.update(FULL_EXTRA)

    # ---- names ------------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.cfg.kind in ("ecfp_desc", "ecfp"):
            names += [f"fp_{i}" for i in range(self.cfg.fp_bits)]
        if self.cfg.kind in ("ecfp_desc", "descriptors"):
            names += [f"desc_{k}" for k in self._desc]
        return names

    # ---- single molecule --------------------------------------------------
    def fingerprint(self, mol: Chem.Mol) -> np.ndarray:
        if self.cfg.use_counts:
            fp = self._gen.GetCountFingerprint(mol)
            arr = np.zeros(self.cfg.fp_bits, dtype=np.float32)
            for idx, count in fp.GetNonzeroElements().items():
                arr[idx] = count
            return arr
        fp = self._gen.GetFingerprint(mol)
        arr = np.zeros(self.cfg.fp_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def descriptors(self, mol: Chem.Mol) -> np.ndarray:
        vals = np.empty(len(self._desc), dtype=np.float32)
        for i, fn in enumerate(self._desc.values()):
            try:
                v = fn(mol)
                vals[i] = 0.0 if v is None or not np.isfinite(v) else float(v)
            except Exception:
                vals[i] = 0.0
        return vals

    def featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.cfg.kind in ("ecfp_desc", "ecfp"):
            parts.append(self.fingerprint(mol))
        if self.cfg.kind in ("ecfp_desc", "descriptors"):
            parts.append(self.descriptors(mol))
        return np.concatenate(parts)

    # ---- batch ------------------------------------------------------------
    def transform(self, smiles: list[str] | np.ndarray) -> FeatureMatrix:
        names = self.feature_names
        n = len(smiles)
        X = np.zeros((n, len(names)), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            try:
                X[i] = self.featurize_mol(mol)
                mask[i] = True
            except Exception as exc:
                LOG.debug("featurization failed for %s: %s", smi, exc)
        if (~mask).any():
            LOG.warning("%d/%d molecules could not be featurized", int((~mask).sum()), n)
        return FeatureMatrix(X=X, feature_names=names, valid_mask=mask)

    # ---- similarity helpers ----------------------------------------------
    def bit_fingerprints(self, smiles: list[str]) -> list:
        """Binary fingerprints for Tanimoto work (clustering, AD, diversity)."""
        out = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(str(smi))
            out.append(self._gen.GetFingerprint(mol) if mol is not None else None)
        return out
