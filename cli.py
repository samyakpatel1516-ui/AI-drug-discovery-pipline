"""Command-line entry point.

    python -m dd_pipeline demo --library-size 6000
    python -m dd_pipeline run --config configs/program.yaml \
        --assay data/potency.csv --library data/vendor.smi
    python -m dd_pipeline screen --model runs/.../models/activity_pIC50.pkl \
        --library data/vendor.smi --out hits.csv
    python -m dd_pipeline config-template > configs/program.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .registry import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dd_pipeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="run the full pipeline on simulated data")
    d.add_argument("--config", type=Path, default=None)
    d.add_argument("--library-size", type=int, default=6000)
    d.add_argument("--n-measured", type=int, default=1800)
    d.add_argument("--run-id", default=None)

    r = sub.add_parser("run", help="run stages 2-3 on your own data")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--assay", type=Path, required=True,
                   help="CSV with smiles,value[,units,relation]")
    r.add_argument("--library", type=Path, required=True, help=".smi/.csv to screen")
    r.add_argument("--admet", nargs="*", default=[], metavar="ENDPOINT=PATH",
                   help="extra endpoint CSVs, e.g. logS=data/sol.csv")
    r.add_argument("--run-id", default=None)

    s = sub.add_parser("screen", help="score a library with an existing model")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--library", type=Path, required=True)
    s.add_argument("--config", type=Path, default=None)
    s.add_argument("--out", type=Path, required=True)

    rp = sub.add_parser("report", help="render a markdown report for an existing run")
    rp.add_argument("run_dir", type=Path)

    sub.add_parser("config-template", help="print a fully populated config to stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))
    log = logging.getLogger("dd_pipeline.cli")

    if args.command == "config-template":
        import yaml
        yaml.safe_dump(PipelineConfig().model_dump(mode="json"), sys.stdout, sort_keys=False)
        return 0

    if args.command == "report":
        from .report import write_report
        out = write_report(args.run_dir)
        log.info("report written to %s", out)
        return 0

    if args.command == "demo":
        from .pipeline import run_demo
        from .report import write_report
        cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
        res = run_demo(cfg, run_id=args.run_id, library_size=args.library_size,
                       n_measured=args.n_measured)
        write_report(res["run"].root)
        log.info("results in %s", res["run"].root)
        return 0

    if args.command == "run":
        from .pipeline import run_from_files
        cfg = PipelineConfig.from_yaml(args.config)
        admet: dict[str, str] = {}
        for item in args.admet:
            if "=" not in item:
                log.error("--admet entries must look like ENDPOINT=PATH, got %r", item)
                return 2
            k, v = item.split("=", 1)
            admet[k] = v
        res = run_from_files(cfg, args.assay, args.library, admet or None, args.run_id)
        log.info("results in %s", res["run"].root)
        return 0

    if args.command == "screen":
        import pandas as pd
        from .chem.featurize import Featurizer
        from .chem.standardize import standardize_table
        from .data.loaders import load_library_smi
        from .models.qsar import EnsembleQSAR

        cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
        model = EnsembleQSAR.load(args.model)
        # The featurizer must match the one the model was trained with, which
        # is why it travels in the model card.
        feat_cfg = cfg.featurizer
        if model.card.featurizer:
            from .config import FeaturizerConfig
            feat_cfg = FeaturizerConfig(**model.card.featurizer)
        featurizer = Featurizer(feat_cfg)

        lib = load_library_smi(args.library)
        lib, _ = standardize_table(lib, smiles_col="smiles")
        fm = featurizer.transform(lib["canonical_smiles"].tolist())
        pred = model.predict(fm.X)
        out = lib.copy()
        out[f"pred_{model.endpoint}"] = pred.mean
        out["pred_std"] = pred.std
        if pred.lower is not None:
            out["pred_lower_90"] = pred.lower
            out["pred_upper_90"] = pred.upper
        out = out.sort_values(f"pred_{model.endpoint}", ascending=False)
        out.to_csv(args.out, index=False)
        log.info("wrote %d scored compounds to %s", len(out), args.out)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
