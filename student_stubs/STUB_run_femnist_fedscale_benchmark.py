#!/usr/bin/env python3
"""STUDENT STUB: larger FL-Control experiments with natural clients + real traces.

Functions marked TODO-STUDENT are intentionally incomplete. Missing workload or
trace files must raise errors; do not fabricate resource distributions.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT / "data" / "student_sources"
OUT_DIR = PROJECT / "results" / "student_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [17,29,43,59,71,89,107,131,167,197]

@dataclass(frozen=True)
class Workload:
    name: str
    natural_pool: int
    clients_per_seed: int
    clients_per_round: int
    rounds: int
    model: str

WORKLOADS = {
    "femnist": Workload("femnist", 3550, 20, 5, 100, "2-conv CNN + 128 FC, 62 classes"),
    "wisdm2019": Workload("wisdm2019", 51, 20, 5, 80, "1D-CNN 64/128 + global pooling, 18 classes"),
}

def deterministic_pair(workload_ids, trace_ids, n, seed):
    rng = np.random.default_rng(seed)
    w = sorted(rng.choice(sorted(workload_ids), size=n, replace=False).tolist())
    t = sorted(rng.choice(sorted(trace_ids), size=n, replace=False).tolist())
    return list(zip(w, t))

def assert_sources(workload):
    fedscale = DATA_DIR / "FedScale"
    missing = []
    if workload == "femnist" and not (DATA_DIR / "leaf").exists():
        missing.append("LEAF/FEMNIST")
    if workload == "wisdm2019" and not any(DATA_DIR.glob("WISDM_Smartphone_Smartwatch_2019*")):
        missing.append("WISDM-2019")
    if not (fedscale / "benchmark" / "dataset" / "data" / "device_info" / "client_device_capacity").exists():
        missing.append("FedScale client_device_capacity")
    if not (fedscale / "benchmark" / "dataset" / "data" / "device_info" / "client_behave_trace").exists():
        missing.append("FedScale client_behave_trace")
    if missing:
        raise FileNotFoundError("Missing real sources: " + ", ".join(missing) + ". No synthetic fallback is allowed.")

def load_natural_workload_clients(workload):
    # TODO-STUDENT:
    # FEMNIST: preserve each LEAF writer as one client.
    # WISDM: preserve each real subject as one client and chronological order.
    raise NotImplementedError("TODO-STUDENT: parse natural workload clients")

def load_fedscale_profiles():
    # TODO-STUDENT:
    # Parse real AIBench/device-capacity, MobiPerf bandwidth, and FLASH
    # availability information from the FedScale dataset tree. Return trace IDs
    # and time-indexed resource/availability records. Do not sample synthetic values.
    raise NotImplementedError("TODO-STUDENT: parse real FedScale environment profiles")

def run(workload):
    assert_sources(workload)
    clients = load_natural_workload_clients(workload)
    traces = load_fedscale_profiles()

    # TODO-STUDENT:
    # For each seed, deterministically pair 20 natural clients with 20 real trace
    # profiles, keep pairing fixed across policies, then execute Random,
    # Resource-only, Loss-only, Oort-style, and FL-Control. Record p95 round time,
    # SLO satisfaction, availability misses, time-to-quality, fairness, and bytes.
    raise NotImplementedError("TODO-STUDENT: connect workloads + traces to FL-Control")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=WORKLOADS, required=True)
    ap.add_argument("--print-config", action="store_true")
    args = ap.parse_args()
    if args.print_config:
        print(json.dumps(asdict(WORKLOADS[args.workload]), indent=2))
        return
    run(args.workload)

if __name__ == "__main__":
    main()
