"""Analog generation for lead optimisation.

Three complementary generators, all of which keep you inside synthesisable
chemistry rather than sampling a latent space and hoping:

  1. `BioisostereEnumerator` -- medicinal-chemistry replacement rules applied
     as RDKit reactions. Produces the kinds of analogs a chemist would propose.
  2. `BRICSRecombiner` -- fragment-swap recombination of a seed set. Good for
     scaffold hopping while reusing known-makeable fragments.
  3. `GeneticOptimizer` -- a scored evolutionary loop over the two above, with
     elitism, novelty pressure and hard filter gates each generation.

The GA is where the multi-parameter score does the work: optimising potency
alone reliably produces greasy, insoluble, hERG-active molecules.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, BRICS, rdFingerprintGenerator

from .synth import sa_score

RDLogger.DisableLog("rdApp.*")
LOG = logging.getLogger("dd_pipeline.chem.enumerate")

# Reaction SMARTS for common, well-precedented replacements. Each is a
# single-step edit, which keeps analogs close to the parent series.
BIOISOSTERE_RULES: dict[str, str] = {
    # Carboxylic acid -> acid bioisosteres (permeability/PK rescue)
    "acid_to_tetrazole": "[c:1][CX3](=O)[OX2H1]>>[c:1]c1nn[nH]n1",
    "acid_to_amide": "[C:1][CX3](=O)[OX2H1]>>[C:1]C(=O)N",
    "acid_to_acylsulfonamide": "[C:1][CX3](=O)[OX2H1]>>[C:1]C(=O)NS(=O)(=O)C",
    # Halogen scans (potency + metabolic blocking)
    "h_to_f_aromatic": "[cH:1]>>[c:1]F",
    "h_to_cl_aromatic": "[cH:1]>>[c:1]Cl",
    "h_to_me_aromatic": "[cH:1]>>[c:1]C",
    "h_to_cn_aromatic": "[cH:1]>>[c:1]C#N",
    "h_to_ocf3": "[cH:1]>>[c:1]OC(F)(F)F",
    # Metabolic soft-spot blocking
    "methyl_to_cf3": "[c:1][CH3]>>[c:1]C(F)(F)F",
    "methyl_to_cyclopropyl": "[c:1][CH3]>>[c:1]C1CC1",
    "ome_to_ocd3_proxy": "[c:1][OX2][CH3]>>[c:1]OC(F)(F)",
    # Ring and linker edits
    "phenyl_to_pyridine": "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1>>[n:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1",
    "amide_to_sulfonamide": "[C:1]C(=O)[NX3:2]>>[C:1]S(=O)(=O)[N:2]",
    "ester_to_amide": "[C:1]C(=O)[OX2][CH3]>>[C:1]C(=O)NC",
    # Solubility handles
    "add_morpholine": "[c:1]Br>>[c:1]N1CCOCC1",
    "add_piperazine": "[c:1]Br>>[c:1]N1CCNCC1",
    "add_hydroxyl": "[CH3:1][c:2]>>[CH2:1](O)[c:2]",
    # Basic-amine removal (hERG mitigation)
    "amine_to_amide": "[CX4:1][NX3H2]>>[CX4:1]NC(=O)C",
    "amine_to_ether": "[CX4:1][NX3H2]>>[CX4:1]OC",
}


def _compile_rules(rules: dict[str, str]) -> dict[str, AllChem.ChemicalReaction]:
    out = {}
    for name, smarts in rules.items():
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is not None:
                rxn.Initialize()
                out[name] = rxn
        except Exception as exc:
            LOG.debug("rule %s failed to compile: %s", name, exc)
    return out


@dataclass
class Analog:
    smiles: str
    parent: str
    origin: str
    generation: int = 0
    score: float = float("nan")
    similarity_to_parent: float = float("nan")


class BioisostereEnumerator:
    def __init__(self, rules: dict[str, str] | None = None, max_per_rule: int = 4) -> None:
        self.rules = _compile_rules(rules or BIOISOSTERE_RULES)
        self.max_per_rule = max_per_rule
        LOG.info("%d bioisostere rules compiled", len(self.rules))

    def enumerate(self, smiles: str, limit: int | None = None) -> list[Analog]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return []
        seen: set[str] = {Chem.MolToSmiles(mol)}
        out: list[Analog] = []
        for name, rxn in self.rules.items():
            try:
                products = rxn.RunReactants((mol,))
            except Exception:
                continue
            kept = 0
            for prods in products:
                if kept >= self.max_per_rule:
                    break
                for p in prods:
                    smi = _sanitize_to_smiles(p)
                    if smi is None or smi in seen:
                        continue
                    seen.add(smi)
                    out.append(Analog(smiles=smi, parent=str(smiles), origin=f"rule:{name}"))
                    kept += 1
            if limit and len(out) >= limit:
                break
        return out[:limit] if limit else out


class BRICSRecombiner:
    """Fragment-based recombination over a seed set."""

    def __init__(self, seeds: Sequence[str], seed: int = 1337) -> None:
        self._seed = seed
        self.rng = random.Random(seed)
        self.fragments = self._decompose(seeds)
        LOG.info("BRICS fragment pool: %d fragments from %d seeds",
                 len(self.fragments), len(seeds))

    @staticmethod
    def _decompose(seeds: Sequence[str]) -> list[Chem.Mol]:
        frags: set[str] = set()
        for smi in seeds:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            try:
                frags.update(BRICS.BRICSDecompose(mol, minFragmentSize=3))
            except Exception:
                continue
        # BRICSDecompose returns a set, and iteration order over a set of
        # strings depends on per-process string hash randomisation. Leaving it
        # unsorted makes the fragment pool -- and therefore the entire
        # generated library -- differ between runs with identical seeds, while
        # looking perfectly reproducible inside any single process. Sorting is
        # the whole fix, and it is not optional for an auditable pipeline.
        mols = [Chem.MolFromSmiles(f) for f in sorted(frags)]
        return [m for m in mols if m is not None]

    def build(self, n: int, max_depth: int = 2) -> list[Analog]:
        if not self.fragments:
            return []
        pool = list(self.fragments)
        self.rng.shuffle(pool)
        # BRICSBuild(scrambleReagents=True) shuffles through the *global*
        # `random` module, not any RNG we pass it. Without seeding that, two
        # runs with the same config produce different libraries -- which
        # silently breaks every provenance guarantee in this package. Seed it,
        # then restore the caller's state so we do not reach outside our scope.
        _prev_state = random.getstate()
        random.seed(self._seed)
        out: list[Analog] = []
        seen: set[str] = set()
        try:
            gen = BRICS.BRICSBuild(pool, scrambleReagents=True, maxDepth=max_depth)
            for mol in gen:
                smi = _sanitize_to_smiles(mol)
                if smi is None or smi in seen:
                    continue
                seen.add(smi)
                out.append(Analog(smiles=smi, parent="", origin="brics"))
                if len(out) >= n:
                    break
        except Exception as exc:
            LOG.warning("BRICS build stopped early: %s", exc)
        finally:
            random.setstate(_prev_state)
        return out


@dataclass
class GAConfig:
    population_size: int = 200
    n_generations: int = 5
    elite_frac: float = 0.15
    mutation_rate: float = 0.8
    max_sa_score: float = 5.0
    min_similarity_to_seed: float = 0.35
    novelty_weight: float = 0.10
    seed: int = 1337


class GeneticOptimizer:
    """Score-guided evolutionary search in explicit chemical space.

    `score_fn` takes a list of SMILES and returns an array of scores to
    maximise -- in practice the MPO desirability from `scoring.desirability`,
    so potency, ADMET and safety are optimised jointly.
    """

    def __init__(
        self,
        score_fn: Callable[[list[str]], np.ndarray],
        config: GAConfig | None = None,
        rules: dict[str, str] | None = None,
        filter_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.score_fn = score_fn
        self.cfg = config or GAConfig()
        self.enumerator = BioisostereEnumerator(rules)
        self.filter_fn = filter_fn
        self.rng = random.Random(self.cfg.seed)
        self._fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.history: list[dict] = []

    # ---- helpers ----------------------------------------------------------
    def _fingerprint(self, smi: str):
        mol = Chem.MolFromSmiles(smi)
        return self._fp.GetFingerprint(mol) if mol is not None else None

    def _similarity(self, a: str, b: str) -> float:
        fa, fb = self._fingerprint(a), self._fingerprint(b)
        if fa is None or fb is None:
            return 0.0
        return float(DataStructs.TanimotoSimilarity(fa, fb))

    def _admissible(self, smi: str, seed_smi: str) -> bool:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        if sa_score(mol) > self.cfg.max_sa_score:
            return False
        if self.filter_fn is not None and not self.filter_fn(smi):
            return False
        if seed_smi and self._similarity(smi, seed_smi) < self.cfg.min_similarity_to_seed:
            return False
        return True

    def _novelty(self, smiles: list[str], reference: list[str]) -> np.ndarray:
        """Mean dissimilarity to the reference set; rewards exploring away
        from what we already have, which keeps the GA from collapsing onto one
        over-optimised molecule."""
        if not reference:
            return np.zeros(len(smiles))
        ref_fps = [f for f in (self._fingerprint(s) for s in reference) if f is not None]
        out = np.zeros(len(smiles))
        for i, smi in enumerate(smiles):
            fp = self._fingerprint(smi)
            if fp is None or not ref_fps:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            out[i] = 1.0 - float(np.mean(sims))
        return out

    # ---- main loop --------------------------------------------------------
    def run(self, seeds: Sequence[str]) -> list[Analog]:
        seeds = [s for s in seeds if Chem.MolFromSmiles(str(s)) is not None]
        if not seeds:
            raise ValueError("no parsable seed molecules")

        population: list[Analog] = [
            Analog(smiles=s, parent=s, origin="seed", generation=0, similarity_to_parent=1.0)
            for s in seeds
        ]
        best_overall: dict[str, Analog] = {a.smiles: a for a in population}

        for gen in range(1, self.cfg.n_generations + 1):
            children: list[Analog] = []
            for parent in population:
                if self.rng.random() > self.cfg.mutation_rate:
                    continue
                for analog in self.enumerator.enumerate(parent.smiles, limit=40):
                    if analog.smiles in best_overall:
                        continue
                    if not self._admissible(analog.smiles, parent.parent or parent.smiles):
                        continue
                    analog.generation = gen
                    analog.parent = parent.smiles
                    analog.similarity_to_parent = self._similarity(analog.smiles, parent.smiles)
                    children.append(analog)

            if not children:
                LOG.info("generation %d produced no admissible children; stopping", gen)
                break

            # Score children, blend in a novelty term, then select.
            smis = [c.smiles for c in children]
            raw = np.asarray(self.score_fn(smis), dtype=float)
            nov = self._novelty(smis, [a.smiles for a in population])
            blended = (1 - self.cfg.novelty_weight) * raw + self.cfg.novelty_weight * nov
            for c, r, b in zip(children, raw, blended):
                c.score = float(r)
                c._blended = float(b)  # type: ignore[attr-defined]

            pool = children + population
            pool.sort(key=lambda a: getattr(a, "_blended", a.score if a.score == a.score else -1e9),
                      reverse=True)
            n_elite = max(1, int(self.cfg.elite_frac * self.cfg.population_size))
            elites = pool[:n_elite]
            remainder = pool[n_elite:self.cfg.population_size * 3]
            self.rng.shuffle(remainder)
            population = elites + remainder[: max(0, self.cfg.population_size - n_elite)]

            for a in population:
                prev = best_overall.get(a.smiles)
                if prev is None or (a.score == a.score and a.score > getattr(prev, "score", -1e9)):
                    best_overall[a.smiles] = a

            scores = np.array([a.score for a in population if a.score == a.score])
            rec = {
                "generation": gen,
                "n_children": len(children),
                "population": len(population),
                "best_score": float(scores.max()) if scores.size else float("nan"),
                "mean_score": float(scores.mean()) if scores.size else float("nan"),
            }
            self.history.append(rec)
            LOG.info(
                "GA gen %d: %d children, best=%.3f mean=%.3f",
                gen, len(children), rec["best_score"], rec["mean_score"],
            )

        out = [a for a in best_overall.values() if a.origin != "seed"]
        out.sort(key=lambda a: (a.score if a.score == a.score else -1e9), reverse=True)
        return out


def _sanitize_to_smiles(mol: Chem.Mol) -> str | None:
    """Products of reaction SMARTS are frequently invalid; this is the gate."""
    try:
        smi = Chem.MolToSmiles(mol)
        m2 = Chem.MolFromSmiles(smi)
        if m2 is None:
            return None
        Chem.SanitizeMol(m2)
        if m2.GetNumHeavyAtoms() < 6:
            return None
        return Chem.MolToSmiles(m2)
    except Exception:
        return None


def enumerate_analogs(
    seeds: Iterable[str], n_per_seed: int = 100, rules: dict[str, str] | None = None
) -> list[Analog]:
    """One-shot rule-based expansion with no scoring (cheap baseline)."""
    enum = BioisostereEnumerator(rules)
    out: list[Analog] = []
    for s in seeds:
        out.extend(enum.enumerate(s, limit=n_per_seed))
    return out
