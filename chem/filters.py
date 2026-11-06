"""Structural alerts and property gates.

Filters run *before* scoring, for two reasons: they are cheap, and a model
asked to rank 10^6 compounds will happily put reactive dyes and aggregators at
the top because those are over-represented among nominal assay actives.

Alerts are advisory, not verdicts. A PAINS hit on a compound with clean
orthogonal data is a flag for the chemist, not an automatic deletion -- so the
reason is always recorded alongside the decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, FilterCatalog, rdMolDescriptors

from ..config import HitDiscoveryConfig

LOG = logging.getLogger("dd_pipeline.chem.filters")

# Covalent-warhead and reactive-group SMARTS worth separating out explicitly:
# some programs want these (targeted covalent inhibitors), most do not.
REACTIVE_SMARTS: dict[str, str] = {
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "alkyl_halide_activated": "[CX4;!$(C(F)(F)F)][Cl,Br,I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "sulfonyl_halide": "S(=O)(=O)[F,Cl,Br]",
    "anhydride": "[CX3](=O)O[CX3](=O)",
    "azide": "[NX2]=[NX2]=[NX1]",
    "nitroso": "[NX2]=[OX1]",
    "thiol": "[SX2H]",
    "hydrazine": "[NX3][NX3]",
    "quinone": "O=C1C=CC(=O)C=C1",
}

_REACTIVE_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in REACTIVE_SMARTS.items()}


@dataclass
class FilterResult:
    passed: bool
    reasons: list[str]

    @property
    def reason_str(self) -> str:
        return ";".join(self.reasons) if self.reasons else ""


class StructuralAlertFilter:
    """RDKit FilterCatalog wrapper (PAINS A/B/C, Brenk, NIH)."""

    def __init__(self, pains: bool = True, brenk: bool = True, nih: bool = False) -> None:
        params = FilterCatalog.FilterCatalogParams()
        cats = []
        if pains:
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            cats.append("PAINS")
        if brenk:
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
            cats.append("BRENK")
        if nih:
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
            cats.append("NIH")
        self.catalog = FilterCatalog.FilterCatalog(params)
        self.categories = cats

    def check(self, mol: Chem.Mol) -> list[str]:
        if not self.categories:
            return []
        matches = self.catalog.GetMatches(mol)
        return [f"alert:{m.GetDescription()}" for m in matches]


def reactive_groups(mol: Chem.Mol) -> list[str]:
    return [
        f"reactive:{name}"
        for name, patt in _REACTIVE_PATTERNS.items()
        if patt is not None and mol.HasSubstructMatch(patt)
    ]


def property_gate(mol: Chem.Mol, cfg: HitDiscoveryConfig) -> list[str]:
    reasons: list[str] = []
    mw = rdMolDescriptors.CalcExactMolWt(mol)
    clogp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)

    if not (cfg.mw_range[0] <= mw <= cfg.mw_range[1]):
        reasons.append(f"mw:{mw:.0f}")
    if not (cfg.clogp_range[0] <= clogp <= cfg.clogp_range[1]):
        reasons.append(f"clogp:{clogp:.2f}")
    if rotb > cfg.max_rotatable_bonds:
        reasons.append(f"rotb:{rotb}")
    if hbd > cfg.max_hbd:
        reasons.append(f"hbd:{hbd}")
    if tpsa > cfg.max_tpsa:
        reasons.append(f"tpsa:{tpsa:.0f}")
    return reasons


def apply_filters(
    df: pd.DataFrame,
    cfg: HitDiscoveryConfig,
    smiles_col: str = "canonical_smiles",
    flag_reactive: bool = True,
) -> pd.DataFrame:
    """Annotate (never silently drop) every compound with pass/fail and reasons."""
    alerts = StructuralAlertFilter(cfg.apply_pains, cfg.apply_brenk, cfg.apply_nih)

    passed: list[bool] = []
    reasons: list[str] = []
    n_alert = n_prop = n_react = 0

    for smi in df[smiles_col].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            passed.append(False)
            reasons.append("unparsable")
            continue
        r: list[str] = []
        a = alerts.check(mol)
        p = property_gate(mol, cfg)
        x = reactive_groups(mol) if flag_reactive else []
        if a:
            n_alert += 1
        if p:
            n_prop += 1
        if x:
            n_react += 1
        r = a + p + x
        passed.append(len(r) == 0)
        reasons.append(";".join(r))

    out = df.copy()
    out["filter_pass"] = passed
    out["filter_reasons"] = reasons
    LOG.info(
        "filters: %d/%d pass (%d structural alerts, %d property gate, %d reactive)",
        int(sum(passed)), len(out), n_alert, n_prop, n_react,
    )
    return out
