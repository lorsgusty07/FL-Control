#!/usr/bin/env python3
"""Diagnose SLO failures by policy from round_metrics_<tag>.csv.

Usage:
    python diagnose_slo_failures.py --tag wisdm2019<suffix>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

def locate_metrics_file(tag: str) -> Path:
    candidates = [
        Path(f"round_metrics_{tag}.csv"),
        Path("results/student_runs") / f"round_metrics_{tag}.csv",
        Path(__file__).resolve().parent / f"round_metrics_{tag}.csv",
        Path(__file__).resolve().parent / "results" / "student_runs" / f"round_metrics_{tag}.csv",
        Path.cwd() / f"round_metrics_{tag}.csv",
        Path.cwd() / "results" / "student_runs" / f"round_metrics_{tag}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find round_metrics_{tag}.csv in current directory or results/student_runs/"
    )

def main():
    ap = argparse.ArgumentParser(description="Diagnose SLO failures from round metrics.")
    ap.add_argument("--tag", required=True, help="Tag suffix of the run (e.g. wisdm2019 or wisdm2019_v3check)")
    ap.add_argument("--out-dir", type=str, default=".", help="Directory to save diagnosis_<tag>.csv (default: .)")
    args = ap.parse_args()

    fpath = locate_metrics_file(args.tag)
    print(f"Reading round metrics from: {fpath}")
    df = pd.read_csv(fpath)

    cols = ["timeout", "dropout", "fallback_used", "slo_met"]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {fpath.name}")

    agg_dict = {col: "mean" for col in cols}
    if "n_elig" in df.columns:
        agg_dict["n_elig"] = "mean"

    diag = df.groupby("policy").agg(agg_dict).reset_index()

    # Sort policies logically if standard names are present
    standard_order = ["random", "fastest", "loss", "oort", "flcontrol", "flcontrol_robust", "fastest_df"]
    diag["_sort_key"] = diag["policy"].apply(lambda p: standard_order.index(p) if p in standard_order else 999)
    diag = diag.sort_values("_sort_key").drop(columns=["_sort_key"])

    print("\n" + "=" * 65)
    print(f"SLO Failure Diagnosis for: {args.tag}")
    print("=" * 65)
    print(diag.to_string(index=False))
    print("=" * 65 + "\n")

    out_path = Path(args.out_dir) / f"diagnosis_{args.tag}.csv"
    diag.to_csv(out_path, index=False)
    print(f"Saved diagnosis table to: {out_path}")

if __name__ == "__main__":
    main()
