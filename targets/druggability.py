"""Druggability and tractability assessment.

"Is this protein a good target?" decomposes into questions with different
evidence types, and conflating them is how programs end up three years into an
undruggable target:

  * **Biological validation** -- does modulating it change disease biology?
  * **Chemical tractability** -- can a small molecule bind it usefully?
  * **Safety** -- what happens elsewhere in the body when you hit it?
  * **Modality fit** -- if small molecules are hopeless, is it a degrader,
    antibody, or oligonucleotide target instead?

This module scores the last three from annotation tables. Nothing here is a
substitute for a structural biologist's opinion; the aim is to rank a few
thousand genes down to a shortlist worth that opinion.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger("dd_pipeline.targets.druggability")

# Precedent by protein family. Derived from which families have yielded
# approved small molecules -- a crude but strongly predictive prior.
FAMILY_PRECEDENT: dict[str, float] = {
    "kinase": 0.95,
    "gpcr": 0.90,
    "nuclear_receptor": 0.85,
    "ion_channel": 0.75,
    "protease": 0.75,
    "phosphodiesterase": 0.80,
    "epigenetic_writer": 0.70,
    "epigenetic_reader": 0.60,
    "transporter": 0.55,
    "oxidoreductase": 0.60,
    "e3_ligase": 0.50,
    "phosphatase": 0.35,
    "transcription_factor": 0.20,
    "scaffold_adaptor": 0.15,
    "structural": 0.10,
    "other_enzyme": 0.50,
    "unknown": 0.30,
}

# Structural-pocket descriptors -> 0-1 contributions. Volume and buriedness
# dominate: a shallow, solvent-exposed site rarely supports nanomolar binding.
POCKET_WEIGHTS: dict[str, float] = {
    "pocket_volume_a3": 0.30,
    "pocket_buriedness": 0.25,
    "pocket_hydrophobicity": 0.20,
    "has_apo_structure": 0.10,
    "has_holo_structure": 0.15,
}


def _norm(x: pd.Series, lo: float, hi: float) -> np.ndarray:
    return np.clip((x.to_numpy(dtype=float) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def pocket_druggability(df: pd.DataFrame) -> pd.Series:
    """Score a binding site from pocket geometry.

    Expected columns (missing ones are skipped, weights renormalised):
      pocket_volume_a3, pocket_buriedness [0-1], pocket_hydrophobicity [0-1],
      has_apo_structure {0,1}, has_holo_structure {0,1}
    """
    score = np.zeros(len(df))
    used = 0.0
    if "pocket_volume_a3" in df:
        # Below ~200 A^3 there is nowhere for a drug-sized ligand to sit;
        # above ~1200 A^3 the site is usually too open to bind tightly.
        v = df["pocket_volume_a3"].to_numpy(dtype=float)
        vol = np.clip((v - 150) / 500, 0, 1) * np.clip((1600 - v) / 600, 0, 1)
        score += POCKET_WEIGHTS["pocket_volume_a3"] * np.clip(vol, 0, 1)
        used += POCKET_WEIGHTS["pocket_volume_a3"]
    for col in ("pocket_buriedness", "pocket_hydrophobicity"):
        if col in df:
            score += POCKET_WEIGHTS[col] * _norm(df[col], 0.0, 1.0)
            used += POCKET_WEIGHTS[col]
    for col in ("has_apo_structure", "has_holo_structure"):
        if col in df:
            score += POCKET_WEIGHTS[col] * df[col].astype(float).clip(0, 1).to_numpy()
            used += POCKET_WEIGHTS[col]
    if used == 0:
        LOG.warning("no pocket columns found; druggability falls back to family precedent")
        return pd.Series(np.full(len(df), np.nan), index=df.index)
    return pd.Series(score / used, index=df.index)


def family_precedent(df: pd.DataFrame, family_col: str = "family") -> pd.Series:
    if family_col not in df:
        return pd.Series(np.full(len(df), FAMILY_PRECEDENT["unknown"]), index=df.index)
    fam = df[family_col].astype(str).str.lower().str.strip()
    return pd.Series(
        fam.map(FAMILY_PRECEDENT).fillna(FAMILY_PRECEDENT["unknown"]).to_numpy(),
        index=df.index,
    )


def ligand_evidence(df: pd.DataFrame) -> pd.Series:
    """Existing chemical matter is the strongest tractability signal there is.

    Columns used if present: n_known_ligands, best_known_pactivity,
    has_approved_drug, has_clinical_candidate, n_patents.
    """
    score = np.zeros(len(df))
    used = 0.0
    if "n_known_ligands" in df:
        n = df["n_known_ligands"].to_numpy(dtype=float)
        score += 0.30 * np.clip(np.log10(n + 1) / 3.0, 0, 1)  # saturates ~1000
        used += 0.30
    if "best_known_pactivity" in df:
        score += 0.25 * _norm(df["best_known_pactivity"], 5.0, 9.0)
        used += 0.25
    if "has_approved_drug" in df:
        score += 0.25 * df["has_approved_drug"].astype(float).clip(0, 1).to_numpy()
        used += 0.25
    if "has_clinical_candidate" in df:
        score += 0.20 * df["has_clinical_candidate"].astype(float).clip(0, 1).to_numpy()
        used += 0.20
    if used == 0:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return pd.Series(score / used, index=df.index)


def safety_score(df: pd.DataFrame) -> pd.Series:
    """Higher is safer. Penalises the liabilities that actually kill programs.

    Columns used if present: is_common_essential, essential_tissues (list or
    comma string), has_germline_lof_phenotype, mouse_ko_lethal,
    paralog_count, expressed_in_heart/liver/cns (0-1 relative expression).
    """
    s = np.ones(len(df))
    if "is_common_essential" in df:
        s -= 0.45 * df["is_common_essential"].astype(float).clip(0, 1).to_numpy()
    if "mouse_ko_lethal" in df:
        s -= 0.25 * df["mouse_ko_lethal"].astype(float).clip(0, 1).to_numpy()
    if "has_germline_lof_phenotype" in df:
        s -= 0.15 * df["has_germline_lof_phenotype"].astype(float).clip(0, 1).to_numpy()
    for tissue, pen in (("expressed_in_heart", 0.15), ("expressed_in_cns", 0.10),
                        ("expressed_in_liver", 0.08)):
        if tissue in df:
            s -= pen * np.clip(df[tissue].to_numpy(dtype=float), 0, 1)
    if "paralog_count" in df:
        # Close paralogs mean selectivity work, and often redundancy that
        # blunts the pharmacology.
        s -= 0.10 * np.clip(df["paralog_count"].to_numpy(dtype=float) / 5.0, 0, 1)
    return pd.Series(np.clip(s, 0.0, 1.0), index=df.index)


def assess_druggability(annotations: pd.DataFrame) -> pd.DataFrame:
    """Combine the tractability views into one annotated table."""
    out = annotations.copy()
    pocket = pocket_druggability(out)
    fam = family_precedent(out)
    # Where there is no structure, fall back to the family prior rather than
    # dropping the gene -- but record which happened.
    out["pocket_druggability"] = pocket
    out["family_precedent"] = fam
    out["druggability_source"] = np.where(pocket.isna(), "family_prior", "structure")
    combined = np.where(pocket.isna(), fam.to_numpy(),
                        0.6 * pocket.fillna(0).to_numpy() + 0.4 * fam.to_numpy())
    out["druggability"] = np.clip(combined, 0, 1)
    out["ligand_evidence"] = ligand_evidence(out)
    out["safety"] = safety_score(out)
    return out


def recommend_modality(row: pd.Series) -> str:
    """Suggest a modality when small molecules look unpromising.

    Cheap to compute and routinely saves a year: a transcription factor with
    no pocket is not a small-molecule target, but it may be an excellent
    degrader or siRNA target.
    """
    drugg = float(row.get("druggability", 0.0) or 0.0)
    fam = str(row.get("family", "unknown")).lower()
    secreted = bool(row.get("is_secreted", False))
    membrane = bool(row.get("is_cell_surface", False))

    if drugg >= 0.55:
        return "small_molecule"
    if secreted or membrane:
        return "antibody"
    if fam in ("transcription_factor", "scaffold_adaptor") or drugg < 0.3:
        if row.get("has_degrader_handle", False):
            return "targeted_degrader"
        return "oligonucleotide"
    return "small_molecule_high_risk"
