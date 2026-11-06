"""Synthetic accessibility.

Uses RDKit's bundled SA_Score (Ertl & Schuffenhauer fragment contributions)
when the Contrib directory is available, with a transparent descriptor-based
fallback so the pipeline never hard-fails on a slim install.

SA score is a rough guide, not a route. For anything you intend to make,
confirm with a retrosynthesis tool or a chemist; the value here is to stop a
generative loop from drifting into unmakeable chemistry.
"""
from __future__ import annotations

import importlib.util
import logging
import sys

from rdkit import Chem
from rdkit.Chem import RDConfig, rdMolDescriptors

LOG = logging.getLogger("dd_pipeline.chem.synth")

_sascorer = None
_tried = False


def _load_sascorer():
    global _sascorer, _tried
    if _tried:
        return _sascorer
    _tried = True
    try:
        path = f"{RDConfig.RDContribDir}/SA_Score/sascorer.py"
        spec = importlib.util.spec_from_file_location("sascorer", path)
        if spec is None or spec.loader is None:
            raise ImportError("sascorer spec unavailable")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sascorer"] = mod
        spec.loader.exec_module(mod)
        _sascorer = mod
        LOG.info("using RDKit SA_Score")
    except Exception as exc:
        LOG.warning("SA_Score unavailable (%s); using descriptor fallback", exc)
        _sascorer = None
    return _sascorer


def _fallback_sa(mol: Chem.Mol) -> float:
    """Monotonic proxy on the same 1-10 scale.

    Penalises the things that actually make synthesis hard: stereocentres,
    fused/spiro/bridged systems, macrocycles and sheer size.
    """
    n_heavy = max(mol.GetNumHeavyAtoms(), 1)
    stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    macro = sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) > 8)
    score = (
        1.0
        + 0.030 * n_heavy
        + 0.45 * stereo
        + 0.60 * spiro
        + 0.50 * bridge
        + 0.20 * max(rings - 3, 0)
        + 1.20 * macro
    )
    return float(min(max(score, 1.0), 10.0))


def sa_score(mol: Chem.Mol | None) -> float:
    """1 (easy) .. 10 (hard). Returns 10.0 for unparsable input."""
    if mol is None:
        return 10.0
    scorer = _load_sascorer()
    if scorer is not None:
        try:
            return float(scorer.calculateScore(mol))
        except Exception:
            pass
    return _fallback_sa(mol)


def sa_score_smiles(smiles: str) -> float:
    return sa_score(Chem.MolFromSmiles(str(smiles)))
