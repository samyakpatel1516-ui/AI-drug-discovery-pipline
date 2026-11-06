"""Structure standardisation.

Almost every reproducibility failure in QSAR traces back to this file being
skipped. Two records of the same compound with different salt forms, charge
states or tautomers become two different training rows, and duplicates leak
across the train/test boundary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
LOG = logging.getLogger("dd_pipeline.chem.standardize")

# Elements we accept. Anything else (metals, boron-free organometallics,
# radioisotopes) is routed out rather than silently featurized.
_ALLOWED = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I",
}


@dataclass
class StandardizerStats:
    n_in: int = 0
    n_unparsable: int = 0
    n_disallowed_element: int = 0
    n_size_filtered: int = 0
    n_duplicate_groups: int = 0
    n_out: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


class Standardizer:
    """Canonicalise a molecule to a single comparable form.

    Order matters: clean up valences, strip salts/solvates, keep the largest
    organic fragment, neutralise where chemically sensible, pick a canonical
    tautomer, then emit canonical SMILES and an InChIKey for deduplication.
    """

    def __init__(self, max_heavy_atoms: int = 70, min_heavy_atoms: int = 6,
                 neutralize: bool = True, canonical_tautomer: bool = True) -> None:
        self.max_heavy_atoms = max_heavy_atoms
        self.min_heavy_atoms = min_heavy_atoms
        self.neutralize = neutralize
        self.canonical_tautomer = canonical_tautomer
        self._salt_remover = SaltRemover.SaltRemover()
        self._lfc = rdMolStandardize.LargestFragmentChooser()
        self._uncharger = rdMolStandardize.Uncharger()
        # Tautomer canonicalisation is the slow step; cap the search.
        params = rdMolStandardize.CleanupParameters()
        params.maxTautomers = 64
        self._tautomer = rdMolStandardize.TautomerEnumerator(params)

    def standardize_mol(self, mol: Chem.Mol | None) -> Chem.Mol | None:
        if mol is None:
            return None
        try:
            mol = rdMolStandardize.Cleanup(mol)
            mol = self._salt_remover.StripMol(mol, dontRemoveEverything=True)
            mol = self._lfc.choose(mol)
            if mol is None or mol.GetNumAtoms() == 0:
                return None
            if self.neutralize:
                mol = self._uncharger.uncharge(mol)
            if self.canonical_tautomer:
                mol = self._tautomer.Canonicalize(mol)
            Chem.SanitizeMol(mol)
            mol.SetProp("_Name", mol.GetProp("_Name") if mol.HasProp("_Name") else "")
            return mol
        except Exception as exc:  # malformed valences, kekulization failures
            LOG.debug("standardization failed: %s", exc)
            return None

    def standardize_smiles(self, smiles: str) -> tuple[str | None, str | None]:
        """Return (canonical_smiles, inchikey) or (None, None)."""
        mol = self.standardize_mol(Chem.MolFromSmiles(smiles))
        if mol is None:
            return None, None
        if not self._element_ok(mol):
            return None, None
        n_heavy = mol.GetNumHeavyAtoms()
        if not (self.min_heavy_atoms <= n_heavy <= self.max_heavy_atoms):
            return None, None
        try:
            key = Chem.MolToInchiKey(mol)
        except Exception:
            key = None
        return Chem.MolToSmiles(mol), key

    def _element_ok(self, mol: Chem.Mol) -> bool:
        return all(a.GetSymbol() in _ALLOWED for a in mol.GetAtoms())


def standardize_table(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    value_col: str | None = None,
    aggregate: str = "median",
    standardizer: Standardizer | None = None,
) -> tuple[pd.DataFrame, StandardizerStats]:
    """Standardise, deduplicate and aggregate an assay table.

    Replicate measurements of the same standardised structure are collapsed
    with `aggregate` (median is robust to the occasional bad well). The spread
    across replicates is kept as `<value>_spread` -- it is the floor on the
    accuracy any model can reach, and is worth reporting next to test error.
    """
    st = standardizer or Standardizer()
    stats = StandardizerStats(n_in=len(df))

    canon: list[str | None] = []
    keys: list[str | None] = []
    for smi in df[smiles_col].astype(str):
        c, k = st.standardize_smiles(smi)
        canon.append(c)
        keys.append(k)

    out = df.copy()
    out["canonical_smiles"] = canon
    out["inchikey"] = keys
    stats.n_unparsable = int(out["canonical_smiles"].isna().sum())
    out = out.dropna(subset=["canonical_smiles"]).copy()

    dedup_key = "inchikey" if out["inchikey"].notna().all() else "canonical_smiles"

    if value_col is not None and value_col in out.columns:
        grp = out.groupby(dedup_key, sort=False)
        stats.n_duplicate_groups = int((grp.size() > 1).sum())
        agg_val = getattr(grp[value_col], aggregate)()
        spread = grp[value_col].agg(lambda s: float(s.max() - s.min()) if len(s) > 1 else 0.0)
        n_rep = grp.size()
        first = grp.first().drop(columns=[value_col], errors="ignore")
        out = first.join(agg_val.rename(value_col)).join(
            spread.rename(f"{value_col}_spread")
        ).join(n_rep.rename("n_replicates"))
        out = out.reset_index()
    else:
        before = len(out)
        out = out.drop_duplicates(subset=[dedup_key]).reset_index(drop=True)
        stats.n_duplicate_groups = before - len(out)

    stats.n_out = len(out)
    LOG.info(
        "standardized %d -> %d rows (%d unparsable, %d duplicate groups collapsed)",
        stats.n_in, stats.n_out, stats.n_unparsable, stats.n_duplicate_groups,
    )
    return out, stats
