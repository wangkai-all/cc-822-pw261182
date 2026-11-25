"""Command-line entry point: run the whole correction chain without the UI.

Examples
--------
# Generate example import CSVs (fictional hindcast/obs/future) into a directory
python -m app.cli generate --out data/imports

# Run cross-validation, calibrate, correct the future and export a product
python -m app.cli run --history data/imports/history.csv \
    --future data/imports/future.csv --out data/runtime/product.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .engine import (CorrectionConfig, CorrectionPipeline,
                     generate_synthetic_dataset, SyntheticConfig, metrics)
from .engine.storage import (export_product, CalibrationStore,
                             import_history, import_future)


def cmd_generate(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    ds = generate_synthetic_dataset(SyntheticConfig(seed=args.seed))
    hist, fut = ds["history"], ds["future"]
    hp = os.path.join(args.out, "history.csv")
    fp = os.path.join(args.out, "future.csv")
    hist[["date", "obs", "hindcast"]].to_csv(hp, index=False)
    fut[["date", "forecast"]].to_csv(fp, index=False)
    print(f"wrote {hp} ({len(hist)} rows) and {fp} ({len(fut)} rows)")
    print("known wet-day freq obs/hindcast: "
          f"{metrics.wet_frequency(hist['obs']):.3f} / "
          f"{metrics.wet_frequency(hist['hindcast']):.3f}")


def cmd_run(args) -> None:
    hist = import_history(args.history)
    fut = import_future(args.future)
    config = CorrectionConfig(
        season_mode=args.season_mode, occ_method=args.occ_method,
        amount_method=args.amount_method, occ_mode=args.occ_mode,
        tail=args.tail, tail_quantile=args.tail_quantile,
        train_years=args.train_years, val_years=args.val_years,
        test_years=args.test_years, stride=args.stride, seed=args.seed)
    pipe = CorrectionPipeline(config)

    cv = pipe.run_cross_validation(hist)
    print(f"\nCross-validation: {len(cv['folds'])} contiguous-year folds")
    for line in cv["plan"].describe():
        print("  " + line)
    print("  recommendation:", cv["recommendation"])

    cal = pipe.fit_final(hist, obs_revision=args.revision,
                         recommendation=cv["recommendation"])
    store = CalibrationStore(args.store)
    store.save(cal)
    print(f"\nCalibration {cal.calibration_id} (fingerprint {cal.fingerprint()})")
    print("  selected methods:", cal.selected_methods)

    product = pipe.apply(fut, cal, occ_mode=args.occ_mode, seed=args.seed)
    truth = fut["future_truth_hidden"].to_numpy() \
        if "future_truth_hidden" in fut.columns else None
    if truth is not None:
        summary = metrics.full_summary(product.raw, product.corrected, truth)
        wf = summary["wet_frequency"]
        print("\nFuture wet-day frequency raw/corrected/truth: "
              f"{wf['raw']:.3f} / {wf['corrected']:.3f} / {wf['obs']:.3f}")
    print(f"  extrapolated beyond training: {int(product.extrapolated.sum())} days")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    path = export_product(product, args.out)
    print(f"\nExported product -> {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="降水预报偏差校正 CLI")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="生成虚构示例 CSV")
    g.add_argument("--out", default="data/imports")
    g.add_argument("--seed", type=int, default=20261125)
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("run", help="交叉验证 + 标定 + 校正 + 导出")
    r.add_argument("--history", required=True)
    r.add_argument("--future", required=True)
    r.add_argument("--out", default="data/runtime/product.csv")
    r.add_argument("--store", default="data/runtime/calibrations")
    r.add_argument("--revision", default="rev-cli-1")
    r.add_argument("--season-mode", default="seasonal",
                   choices=["annual", "seasonal"])
    r.add_argument("--occ-method", default="auto",
                   choices=["auto", "plain", "trend"])
    r.add_argument("--amount-method", default="auto", choices=["auto", "qm", "qdm"])
    r.add_argument("--occ-mode", default="deterministic",
                   choices=["deterministic", "stochastic"])
    r.add_argument("--tail", default="extrapolate",
                   choices=["clip", "extrapolate"])
    r.add_argument("--tail-quantile", type=float, default=0.85)
    r.add_argument("--train-years", type=int, default=6)
    r.add_argument("--val-years", type=int, default=3)
    r.add_argument("--test-years", type=int, default=3)
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--seed", type=int, default=None)
    r.set_defaults(func=cmd_run)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
