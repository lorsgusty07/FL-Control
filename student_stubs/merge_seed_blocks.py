#!/usr/bin/env python3
"""Merge seed blocks produced by STUB_run_femnist_fedscale_benchmark.py.

Example (20 seeds run in 4 blocks of 5):
    python student_stubs/merge_seed_blocks.py --workload wisdm2019 --inputs _b1 _b2 _b3 _b4

Writes (in the same results directory):
    seed_summary_<workload><out>.csv, round_metrics_<workload><out>.csv,
    aggregate_<workload><out>.csv, calibration_<workload><out>.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from STUB_run_femnist_fedscale_benchmark import OUT_DIR, POLICIES, aggregate_seed_summaries  # noqa: E402


def _load_blocks(prefix: str, workload: str, suffixes: list[str]) -> pd.DataFrame:
    frames = []
    for s in suffixes:
        path = OUT_DIR / f"{prefix}_{workload}{s}.csv"
        if not path.exists():
            print(f"  [warn] missing {path.name}")
            continue
        df = pd.read_csv(path)
        df["block"] = s or "(none)"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {prefix} files found for {workload} with suffixes {suffixes}")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=["femnist", "wisdm2019"])
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Out-suffixes of the blocks to merge (use '' for the unsuffixed run)")
    ap.add_argument("--out-suffix", default="_merged")
    args = ap.parse_args()

    # ---- seed summaries
    seeds = _load_blocks("seed_summary", args.workload, args.inputs)
    dup = seeds.duplicated(subset=["seed", "policy"], keep="last")
    if dup.any():
        dups = sorted(seeds.loc[dup, "seed"].unique().tolist())
        print(f"  [warn] duplicated seeds across blocks {dups}: keeping the last block")
        seeds = seeds[~dup]
    n_pol = seeds.groupby("seed")["policy"].nunique()
    incomplete = n_pol[n_pol < len(POLICIES)].index.tolist()
    if incomplete:
        print(f"  [warn] seeds with missing policies (excluded): {incomplete}")
        seeds = seeds[~seeds["seed"].isin(incomplete)]

    # Consistency check: all blocks must share the same SLO quantile and round budget
    calib = {}
    quantiles = set()
    for s in args.inputs:
        cpath = OUT_DIR / f"calibration_{args.workload}{s}.json"
        if cpath.exists():
            block = json.loads(cpath.read_text())
            calib.update(block)
            quantiles |= {v["slo_quantile"] for v in block.values()}
    if len(quantiles) > 1:
        raise ValueError(f"Blocks use different SLO quantiles {sorted(quantiles)}; merge them separately.")
    if "rounds_completed" in seeds.columns and seeds["rounds_completed"].nunique() > 1:
        print(f"  [warn] different rounds_completed across seeds: "
              f"{sorted(seeds['rounds_completed'].unique().tolist())}")

    out = args.out_suffix
    seeds = seeds.drop(columns=["block"]).sort_values(["seed", "policy"])
    seeds.to_csv(OUT_DIR / f"seed_summary_{args.workload}{out}.csv", index=False)
    aggregate_seed_summaries(seeds).to_csv(OUT_DIR / f"aggregate_{args.workload}{out}.csv", index=False)
    (OUT_DIR / f"calibration_{args.workload}{out}.json").write_text(json.dumps(calib, indent=2))

    # ---- round metrics (optional, can be large)
    try:
        rounds = _load_blocks("round_metrics", args.workload, args.inputs)
        rounds = rounds[rounds["seed"].isin(seeds["seed"].unique())]
        rounds = rounds.drop_duplicates(subset=["seed", "policy", "round"], keep="last")
        rounds.drop(columns=["block"]).to_csv(OUT_DIR / f"round_metrics_{args.workload}{out}.csv", index=False)
    except FileNotFoundError as e:
        print(f"  [warn] {e}")

    print(f"Merged {seeds['seed'].nunique()} seeds x {seeds['policy'].nunique()} policies "
          f"-> {OUT_DIR / f'aggregate_{args.workload}{out}.csv'}")


if __name__ == "__main__":
    main()
