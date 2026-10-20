#!/usr/bin/env python3
# Elegir margin/drop_max aquí y CONGELARLOS antes de correr las seeds de reporte 0-9.
"""Development parameter sweep for margin and drop_max.

Sweeps safety margin in {0.0, 0.05, 0.10, 0.15, 0.25, 0.40} and drop_max in {1.0, 0.5}
on development seeds 100, 101, 102, 103, 104 (disjoint from evaluation seeds).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

# Support importing benchmark modules whether run from repo root or student_stubs/
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "student_stubs"))
from STUB_run_femnist_fedscale_benchmark import run, OUT_DIR  # noqa: E402

DEV_SEEDS = [100, 101, 102, 103, 104]
MARGINS = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
DROP_MAXES = [1.0, 0.5]
SWEEP_POLICIES = ["fastest", "fastest_df", "flcontrol", "flcontrol_robust"]

def main():
    ap = argparse.ArgumentParser(description="Sweep margin and drop_max on development seeds.")
    ap.add_argument("--workload", default="wisdm2019", choices=["wisdm2019", "femnist"])
    ap.add_argument("--rounds", type=int, default=None, help="Override rounds (default: workload rounds)")
    ap.add_argument("--warmup-rounds", type=int, default=10, help="Warmup FedAvg rounds (default: 10)")
    ap.add_argument("--resume", action="store_true", help="Skip runs already present in checkpoints")
    args = ap.parse_args()

    summary_rows = []

    for drop_max in DROP_MAXES:
        for margin in MARGINS:
            suffix = f"_dev_m{margin}_d{drop_max}"
            print(f"\n{'='*75}")
            print(f"Executing sweep configuration: margin={margin}, drop_max={drop_max} (suffix={suffix})")
            print(f"{'='*75}")

            run(
                workload_name=args.workload,
                seeds=DEV_SEEDS,
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
                margin=margin,
                drop_max=drop_max,
                policies=SWEEP_POLICIES,
                out_suffix=suffix,
                resume=args.resume
            )

            # Read aggregate results for this run
            agg_candidates = [
                OUT_DIR / f"aggregate_{args.workload}{suffix}.csv",
                Path(f"aggregate_{args.workload}{suffix}.csv"),
                Path.cwd() / f"aggregate_{args.workload}{suffix}.csv"
            ]
            agg_path = next((p for p in agg_candidates if p.exists()), None)

            if agg_path:
                df_agg = pd.read_csv(agg_path, header=[0, 1])
                # Flatten multi-level columns
                df_agg.columns = [f"{c0}_{c1}".strip("_") if c1 else c0 for c0, c1 in df_agg.columns]
                pol_col = [c for c in df_agg.columns if "policy" in c][0]

                for _, r in df_agg.iterrows():
                    pol = r[pol_col]
                    summary_rows.append({
                        "margin": margin,
                        "drop_max": drop_max,
                        "policy": pol,
                        "slo_rate": r.get("slo_rate_mean", r.get("slo_latency_rate_mean", float("nan"))),
                        "final_f1": r.get("final_f1_mean", float("nan")),
                        "fairness_selection_norm": r.get("fairness_selection_norm_mean", float("nan")),
                        "p95_round_time": r.get("p95_round_time_mean", float("nan")),
                    })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        print("\n" + "=" * 80)
        print("DEVELOPMENT SWEEP SUMMARY")
        print("=" * 80)
        print(df_summary.to_string(index=False))
        print("=" * 80 + "\n")

        out_file = Path("sweep_summary.csv")
        df_summary.to_csv(out_file, index=False)
        print(f"Sweep summary written to {out_file.resolve()}")

if __name__ == "__main__":
    main()
