#!/usr/bin/env python3
"""STUDENT BENCHMARK: FL-Control experiments with natural clients + real traces.

Provides complete implementations for:
- Workload client loading (LEAF/FEMNIST natural writers & WISDM-2019 subjects)
- Real FedScale environment trace parsing (compute, communication bandwidth, FLASH availability)
- Models: 2-conv CNN + 128 FC (FEMNIST) and 1D-CNN 64/128 + global pooling (WISDM-2019)
- Policies: Random, Resource-only (Fastest), Loss-only, Oort-style, and FL-Control
- Output metrics: p95 round time, SLO satisfaction, availability misses, time-to-quality, fairness, and bytes
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse
import bisect
import copy
import glob
import itertools
import json
import os
import pickle
import zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score

# Detect GPU device (T4 on Google Colab or fallback to CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    # Enable cuDNN benchmark for optimized conv operations
    torch.backends.cudnn.benchmark = True
else:
    torch.set_num_threads(1)

# Dynamically locate DATA_DIR and OUT_DIR across local and Colab environments
def resolve_directories():
    here = Path(__file__).resolve()
    candidate_roots = [
        here.parents[2],      # standard local layout: repo_root/code/student_stubs/...
        here.parents[1],      # Colab direct clone: repo_root/student_stubs/...
        Path("/content"),     # Colab workspace root
        Path.cwd().parent,
        Path.cwd()
    ]
    data_dir = None
    for r in candidate_roots:
        if (r / "data" / "student_sources").exists():
            data_dir = r / "data" / "student_sources"
            out_dir = r / "results" / "student_runs"
            break

    if data_dir is None:
        # Fallback to local default
        data_dir = here.parents[2] / "data" / "student_sources"
        out_dir = here.parents[2] / "results" / "student_runs"

    out_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, out_dir

DATA_DIR, OUT_DIR = resolve_directories()
# 20 fixed seeds: the first 10 are the original ones (results remain comparable)
SEEDS = [17, 29, 43, 59, 71, 89, 107, 131, 167, 197,
         223, 251, 283, 311, 347, 379, 409, 443, 479, 509]

# Seed-specific SLO deadline rule (Sec. III-B): B = q-quantile of calibration costs.
# The same q is used for every workload; ablate with --slo-quantile 0.50 / 0.90.
SLO_QUANTILE = 0.75
POLICIES = ["random", "fastest", "loss", "oort", "flcontrol"]

@dataclass(frozen=True)
class Workload:
    name: str
    natural_pool: int
    clients_per_seed: int
    clients_per_round: int
    rounds: int
    model: str

WORKLOADS = {
    "femnist": Workload("femnist", 3550, 20, 5, 300, "2-conv CNN + 128 FC, 62 classes"),
    "wisdm2019": Workload("wisdm2019", 51, 20, 5, 300, "1D-CNN 64/128 + global pooling, 18 classes"),
}

# ==========================================
# Model Architectures
# ==========================================
class FemnistCNN(nn.Module):
    """2-conv CNN + 128 FC for 62 classes (FEMNIST)."""
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Wisdm1DCNN(nn.Module):
    """1D-CNN 64/128 + global pooling for 18 classes (WISDM 2019)."""
    def __init__(self, in_channels=3, num_classes=18):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Expected input shape: (batch_size, in_channels, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.fc(x)

def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

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

# ==========================================
# Data Loaders (Real natural sources)
# ==========================================
def load_natural_workload_clients(workload: str):
    """Load natural clients preserving each real subject/writer."""
    clients = {}
    if workload == "femnist":
        leaf_data_dir = DATA_DIR / "leaf" / "data" / "femnist" / "data"
        train_files = sorted(glob.glob(str(leaf_data_dir / "train" / "*.json")))
        test_files = sorted(glob.glob(str(leaf_data_dir / "test" / "*.json")))

        if not train_files:
            raise FileNotFoundError(f"No LEAF FEMNIST train JSON files found in {leaf_data_dir / 'train'}")

        # Parse LEAF json files
        for fpath in train_files:
            with open(fpath, "r") as f:
                data = json.load(f)
                for user in data["users"]:
                    u_data = data["user_data"][user]
                    X = np.array(u_data["x"], dtype=np.float32)
                    y = np.array(u_data["y"], dtype=np.int64)
                    clients[user] = {"train_X": X, "train_y": y, "test_X": None, "test_y": None}

        for fpath in test_files:
            with open(fpath, "r") as f:
                data = json.load(f)
                for user in data["users"]:
                    if user in clients:
                        u_data = data["user_data"][user]
                        clients[user]["test_X"] = np.array(u_data["x"], dtype=np.float32)
                        clients[user]["test_y"] = np.array(u_data["y"], dtype=np.int64)

        # Fallback split if test not partitioned
        for u, d in clients.items():
            if d["test_X"] is None or len(d["test_X"]) == 0:
                n = len(d["train_y"])
                n_tr = max(1, int(0.8 * n))
                d["test_X"] = d["train_X"][n_tr:]
                d["test_y"] = d["train_y"][n_tr:]
                d["train_X"] = d["train_X"][:n_tr]
                d["train_y"] = d["train_y"][:n_tr]

    elif workload == "wisdm2019":
        # Check zip archive or extracted folder
        zip_candidates = list(DATA_DIR.glob("WISDM_Smartphone_Smartwatch_2019*.zip"))
        extracted_dirs = list(DATA_DIR.glob("WISDM_Smartphone_Smartwatch_2019*"))
        
        extracted_dir = None
        for cand in extracted_dirs:
            if cand.is_dir():
                extracted_dir = cand
                break

        if extracted_dir is None and zip_candidates:
            target_zip = zip_candidates[0]
            extracted_dir = DATA_DIR / "WISDM_extracted"
            if not extracted_dir.exists():
                print(f"Extracting {target_zip}...")
                with zipfile.ZipFile(target_zip, 'r') as z:
                    z.extractall(extracted_dir)

        if extracted_dir is None or not extracted_dir.exists():
            raise FileNotFoundError("Could not find WISDM-2019 directory or zip archive.")

        # Find raw text/csv files recursively
        raw_files = sorted(glob.glob(str(extracted_dir / "**" / "*.txt"), recursive=True))
        if not raw_files:
            raw_files = sorted(glob.glob(str(extracted_dir / "**" / "*.csv"), recursive=True))

        print(f"Found {len(raw_files)} WISDM raw data files in {extracted_dir}")
        activity_map = {}

        for fpath in raw_files:
            fname = Path(fpath).stem
            # Filter out documentation/readme files
            if fname.lower() in ["readme", "license", "description"] or fname.startswith("."):
                continue

            # Read lines directly to handle trailing semicolons cleanly
            rows_X = []
            rows_y = []
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_idx, line in enumerate(f):
                        if line_idx >= 5000:
                            break
                        line = line.strip().rstrip(";")
                        if not line:
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 6:
                            sub_id = parts[0]
                            act = parts[1]
                            try:
                                x_val = float(parts[3])
                                y_val = float(parts[4])
                                z_val = float(parts[5])
                            except ValueError:
                                continue
                            if act not in activity_map:
                                activity_map[act] = len(activity_map)
                            rows_X.append([x_val, y_val, z_val])
                            rows_y.append(activity_map[act])
                            subject_id = sub_id
            except Exception as e:
                continue

            if len(rows_X) >= 128:
                mat = np.array(rows_X, dtype=np.float32)
                y_vec = np.array(rows_y, dtype=np.int64)
                win_size = 128
                n_wins = len(mat) // win_size
                if n_wins > 0:
                    X_wins = mat[:n_wins * win_size].reshape(n_wins, 3, win_size)
                    y_wins = y_vec[:n_wins * win_size:win_size]
                    if subject_id not in clients:
                        clients[subject_id] = {"X": [], "y": []}
                    clients[subject_id]["X"].append(X_wins)
                    clients[subject_id]["y"].append(y_wins)

        for sid in list(clients.keys()):
            if clients[sid]["X"]:
                X = np.concatenate(clients[sid]["X"], axis=0)
                y = np.concatenate(clients[sid]["y"], axis=0)
                n_tr = int(0.8 * len(y))
                clients[sid] = {
                    "train_X": X[:n_tr], "train_y": y[:n_tr],
                    "test_X": X[n_tr:], "test_y": y[n_tr:]
                }

    if len(clients) == 0:
        raise RuntimeError(f"No clients successfully loaded for workload: {workload}")

    return clients

# ==========================================
# Real FedScale Trace Profiles Loader
# ==========================================
class FedScaleProfile:
    def __init__(self, trace_id, compute_speed, bandwidth, avail_trace=None):
        self.trace_id = trace_id
        self.compute_speed = float(compute_speed)       # ms per sample
        self.bandwidth = float(bandwidth)               # KB/s
        self.avail_trace = avail_trace

    def _trace(self):
        """Return (actives, inactives, finish_time) or None if the client is always available.

        FLASH/FedScale format: aligned lists where the client is active on
        [active[k], inactive[k]]; the trace repeats with period finish_time.
        If the last active segment has no closing timestamp, it lasts until finish_time.
        """
        if not self.avail_trace:
            return None
        finish_time = float(self.avail_trace.get("finish_time", 86400 * 7))
        actives = self.avail_trace.get("active", [])
        inactives = self.avail_trace.get("inactive", [])
        if finish_time <= 0 or not actives:
            return None
        return actives, inactives, finish_time

    @staticmethod
    def _segment_end(inactives, k, finish_time):
        return float(inactives[k]) if k < len(inactives) else finish_time

    def is_active(self, cur_time: float) -> bool:
        """Evaluate FLASH real client availability trace at given wall-clock time (stateless)."""
        tr = self._trace()
        if tr is None:
            return True
        actives, inactives, finish_time = tr
        norm_time = cur_time % finish_time
        # Last segment that started at or before norm_time
        k = bisect.bisect_right(actives, norm_time) - 1
        if k < 0:
            return False
        return norm_time <= self._segment_end(inactives, k, finish_time)

    def first_inactive_time(self, start: float, end: float):
        """Exact check over [start, end]: return the first absolute time the client is
        unavailable, or None if it stays available during the whole interval."""
        tr = self._trace()
        if tr is None:
            return None
        actives, inactives, finish_time = tr
        t = float(start)
        for _ in range(100000):  # hard guard against malformed traces
            if t > end:
                return None
            norm_time = t % finish_time
            k = bisect.bisect_right(actives, norm_time) - 1
            if k < 0:
                return t
            seg_end = self._segment_end(inactives, k, finish_time)
            if norm_time > seg_end:
                return t
            abs_seg_end = t + (seg_end - norm_time)
            if abs_seg_end >= end:
                return None
            # Step just past the segment end (handles contiguous segments and trace wrap-around)
            t = abs_seg_end + 1e-6
        return None

    def is_active_interval(self, start: float, end: float) -> bool:
        return self.first_inactive_time(start, end) is None

    def estimate_round_time(self, num_samples: int, epochs: int, model_bytes: int) -> float:
        """Compute execution time based on real AIBench latency & MobiPerf bandwidth."""
        # Compute: 3x inference time per sample (1 forward + 2 backward equivalent)
        comp_sec = (3.0 * num_samples * epochs * self.compute_speed) / 1000.0
        # Comm: Download + Upload over real bandwidth
        comm_sec = (2.0 * model_bytes) / (max(self.bandwidth, 1.0) * 1024.0)
        return comp_sec + comm_sec

def load_fedscale_profiles():
    """Parse real FedScale device capacity and FLASH behavior traces."""
    fedscale_dir = DATA_DIR / "FedScale" / "benchmark" / "dataset" / "data" / "device_info"
    cap_file = fedscale_dir / "client_device_capacity"
    behave_file = fedscale_dir / "client_behave_trace"

    with open(cap_file, "rb") as f:
        capacity_dict = pickle.load(f)

    with open(behave_file, "rb") as f:
        behave_dict = pickle.load(f)

    profiles = {}
    behave_keys = list(behave_dict.keys())
    for tid, speed in capacity_dict.items():
        avail = behave_dict[behave_keys[int(tid) % len(behave_dict)]] if behave_keys else None
        profiles[tid] = FedScaleProfile(
            trace_id=tid,
            compute_speed=speed.get("computation", 150.0),
            bandwidth=speed.get("communication", 2000.0),
            avail_trace=avail
        )
    return profiles

# ==========================================
# FL Training Utilities
# ==========================================
def get_model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())

def local_loss(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
        logits = model(xt)
        return float(F.cross_entropy(logits, yt).item())

def adaptive_local_train(global_model: nn.Module, X: np.ndarray, y: np.ndarray,
                         seed: int, rel_reduction=0.20, max_epochs=10, lr=0.01):
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()
    gen = torch.Generator(device=DEVICE if DEVICE.type == "cuda" else "cpu").manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y, dtype=torch.long, device=DEVICE)
    n = len(yt)
    start_loss = local_loss(model, X, y)
    target_loss = start_loss * (1.0 - rel_reduction)
    epochs = 0

    for ep in range(1, max_epochs + 1):
        order = torch.randperm(n, generator=gen, device=DEVICE)
        batch_size = max(1, min(16, n))
        for idx in order.split(batch_size):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        epochs = ep
        if local_loss(model, X, y) <= target_loss:
            break

    final_loss = local_loss(model, X, y)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, epochs, start_loss, final_loss

def fedavg(states: list[dict], weights: list[float]) -> dict:
    total_weight = float(sum(weights))
    avg_state = {}
    for k in states[0]:
        avg_state[k] = sum(s[k] * (w / total_weight) for s, w in zip(states, weights))
    return avg_state

def evaluate_global(model: nn.Module, test_X: np.ndarray, test_y: np.ndarray):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(test_X, dtype=torch.float32, device=DEVICE)
        logits = model(xt)
        preds = logits.argmax(1).cpu().numpy()
    acc = accuracy_score(test_y, preds)
    f1 = f1_score(test_y, preds, average="macro", zero_division=0)
    return acc, f1

# ==========================================
# Client Selection Policies
# ==========================================
def select_cohort(policy: str, candidates: list[str], losses: dict[str, float],
                  cost_preds: dict[str, float], deadline: float, k: int,
                  rng: np.random.Generator, oort_alpha: float = 1.0,
                  sample_sizes: dict[str, int] = None) -> tuple[list[str], int]:
    """Select cohort of k clients according to policy.
    Returns: (selected_cohort, fallback_used: int [0 or 1])
    """
    if policy == "random":
        return list(rng.choice(candidates, size=k, replace=False)), 0

    if policy == "fastest":
        return sorted(candidates, key=lambda u: (cost_preds[u], u))[:k], 0

    # Effective learning utility: loss weighted by local sample size |D_i|
    utilities = {u: losses[u] * (sample_sizes[u] if sample_sizes else 1.0) for u in candidates}

    if policy == "loss":
        return sorted(candidates, key=lambda u: (utilities[u], -ord(str(u)[0])), reverse=True)[:k], 0

    if policy == "oort":
        # Oort utility: sample_weighted_loss * min(1, (SLO / duration)^alpha)
        def oort_utility(u):
            duration = max(cost_preds[u], 1e-4)
            speed_penalty = min(1.0, (deadline / duration) ** oort_alpha)
            return utilities[u] * speed_penalty

        return sorted(candidates, key=lambda u: oort_utility(u), reverse=True)[:k], 0

    if policy == "flcontrol":
        # Constrained optimization: max sum(utilities) s.t. max(cost) <= deadline
        feasible = []
        for comb in itertools.combinations(candidates, k):
            max_c = max(cost_preds[u] for u in comb)
            if max_c <= deadline + 1e-9:
                feasible.append((sum(utilities[u] for u in comb), -max_c, comb))
        if feasible:
            return list(max(feasible)[2]), 0

        # If infeasible, fallback: minimize predicted violation then maximize learning utility
        # (Problem 8: documented via fallback_used flag)
        cand = []
        for comb in itertools.combinations(candidates, k):
            max_c = max(cost_preds[u] for u in comb)
            cand.append((-max_c, sum(utilities[u] for u in comb), comb))
        return list(max(cand)[2]), 1

    raise ValueError(f"Unknown policy: {policy}")

# ==========================================
# Main Experiment Execution
# ==========================================
def jain_index(values) -> float:
    """Jain's fairness index; 0.0 when every value is zero."""
    arr = np.asarray(list(values), dtype=float)
    total = arr.sum()
    if arr.size == 0 or total <= 0:
        return 0.0
    return float(total ** 2 / (arr.size * (arr ** 2).sum() + 1e-9))

# Per-seed metrics aggregated as mean/std across seeds.
# time_to_quality_sec is NaN when the target was never reached, so its mean is
# conditional on reached_target == 1; the mean of reached_target is the fraction reached.
AGG_METRICS = [
    "final_f1", "p95_round_time", "p95_round_time_uncapped",
    "slo_rate", "slo_latency_rate", "mean_jain_round",
    "availability_misses", "deadline_timeouts", "flcontrol_fallback_rounds",
    "time_to_quality_sec", "reached_target",
    "fairness_index", "fairness_selection", "fairness_selection_norm",
    "fairness_participation", "fairness_participation_norm",
    "total_bytes", "total_wall_clock_sec", "slo_deadline_sec",
]

def aggregate_seed_summaries(df_seeds: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in AGG_METRICS if c in df_seeds.columns]
    return df_seeds.groupby("policy").agg({c: ["mean", "std"] for c in cols}).reset_index()

def _append_csv(rows: list[dict], path: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)

def common_warmup(init_model: nn.Module, paired_clients: list[str],
                  paired_profiles: dict, clients_pool: dict,
                  clients_per_round: int, seed: int,
                  warmup_rounds: int = 10) -> nn.Module:
    """Pre-train common model for warmup_rounds (Problem 1 Fix).
    Calibrating from this warm model yields realistic LocalSGD workloads
    rather than cold-model transient artifacts. All policies compete starting
    from this identical warm_model checkpoint.
    """
    warm_model = copy.deepcopy(init_model).to(DEVICE)
    warm_rng = np.random.default_rng(seed + 9999)
    cur_clock = 0.0
    model_size = get_model_size_bytes(warm_model)

    for wr in range(1, warmup_rounds + 1):
        avail = [w for w in paired_clients if paired_profiles[w].is_active(cur_clock)]
        if len(avail) < clients_per_round:
            cur_clock += 30.0
            continue

        cohort = list(warm_rng.choice(avail, size=clients_per_round, replace=False))
        states, weights, times = [], [], []

        for j, w in enumerate(cohort):
            c = clients_pool[w]
            st, ep, _, _ = adaptive_local_train(
                warm_model, c["train_X"], c["train_y"],
                seed=seed * 50000 + wr * 100 + j,
                max_epochs=10
            )
            states.append(st)
            weights.append(len(c["train_y"]))
            t = paired_profiles[w].estimate_round_time(len(c["train_y"]), ep, model_size)
            times.append(t)

        if states:
            warm_model.load_state_dict(fedavg(states, weights))
        cur_clock += max(times) if times else 30.0

    return warm_model

def run(workload_name: str, seeds=None, rounds=None, slo_quantile: float = SLO_QUANTILE,
        f1_min: float = None, jain_min: float = 0.50, warmup_rounds: int = 10,
        out_suffix: str = "", resume: bool = False):
    assert_sources(workload_name)
    cfg = WORKLOADS[workload_name]
    if rounds:
        cfg = replace(cfg, rounds=int(rounds))
    seeds = [int(s) for s in (seeds or SEEDS)]

    # Problem 2 & 3: realistic reachable thresholds
    # F1_MIN default: 0.45 for WISDM-2019, 0.20 for FEMNIST
    f1_min = float(f1_min) if f1_min is not None else (0.45 if workload_name == "wisdm2019" else 0.20)
    jain_min = float(jain_min)

    clients_pool = load_natural_workload_clients(workload_name)
    profiles_pool = load_fedscale_profiles()

    workload_ids = list(clients_pool.keys())
    trace_ids = list(profiles_pool.keys())

    # Output files (one set per block, identified by --out-suffix)
    tag = f"{workload_name}{out_suffix}"
    rounds_ckpt = OUT_DIR / f"round_metrics_{tag}.csv"
    seeds_ckpt = OUT_DIR / f"seed_summary_{tag}.csv"
    calib_path = OUT_DIR / f"calibration_{tag}.json"
    agg_path = OUT_DIR / f"aggregate_{tag}.csv"
    config_path = OUT_DIR / f"experiment_config_{tag}.json"

    done_seeds: set[int] = set()
    calibration: dict = {}
    if resume and seeds_ckpt.exists():
        prev = pd.read_csv(seeds_ckpt)
        n_pol = prev.groupby("seed")["policy"].nunique()
        done_seeds = {int(s) for s in n_pol[n_pol == len(POLICIES)].index}
        if calib_path.exists():
            calibration = json.loads(calib_path.read_text())
        print(f"Resuming {tag}: seeds already complete -> {sorted(done_seeds)}")
    elif not resume:
        for p in (rounds_ckpt, seeds_ckpt, calib_path, agg_path):
            if p.exists():
                p.unlink()

    # Benchmark-wide common quality threshold for wall-clock Time-to-Quality (TTQ)
    target_f1 = 0.20 if workload_name == "femnist" else 0.50

    for seed in seeds:
        if seed in done_seeds:
            print(f"\n>>> Skipping seed {seed} (already in checkpoint)")
            continue
        print(f"\n>>> Running Seed {seed} for Workload {workload_name}")
        # Deterministically pair 20 natural clients with 20 real trace profiles
        pairing = deterministic_pair(workload_ids, trace_ids, cfg.clients_per_seed, seed)
        paired_clients = [w for w, _ in pairing]
        paired_profiles = {w: profiles_pool[t] for w, t in pairing}

        # Build pooled test set across the 20 clients
        val_X_list, val_y_list = [], []
        for w in paired_clients:
            if clients_pool[w]["test_X"] is not None and len(clients_pool[w]["test_X"]) > 0:
                val_X_list.append(clients_pool[w]["test_X"])
                val_y_list.append(clients_pool[w]["test_y"])
        val_X = np.concatenate(val_X_list, axis=0)
        val_y = np.concatenate(val_y_list, axis=0)

        # Common model initialization
        seed_all(seed)
        num_classes = 62 if workload_name == "femnist" else 18
        init_model = (FemnistCNN(num_classes) if workload_name == "femnist" else Wisdm1DCNN(num_classes=num_classes)).to(DEVICE)
        model_size = get_model_size_bytes(init_model)

        # Problem 1 Fix: Common warmup of 10 rounds before calibration
        print(f"Seed {seed}: Executing {warmup_rounds} common warmup rounds to initialize warm model...")
        warm_model = common_warmup(init_model, paired_clients, paired_profiles,
                                   clients_pool, cfg.clients_per_round, seed,
                                   warmup_rounds=warmup_rounds)

        # Baseline calibration on warm_model to determine the realistic seed-specific SLO envelope
        cal_costs = []
        for w in paired_clients:
            c = clients_pool[w]
            st, ep, _, _ = adaptive_local_train(warm_model, c["train_X"], c["train_y"], seed, max_epochs=10)
            est_t = paired_profiles[w].estimate_round_time(len(c["train_y"]), ep, model_size)
            cal_costs.append(est_t)

        # Seed-specific deadline: B = q-quantile of calibration costs (same q for all workloads).
        # B stays fixed for the whole run (no recalibration), as defined in Sec. III.
        slo_deadline = float(np.quantile(cal_costs, slo_quantile))
        print(f"Seed {seed} SLO Deadline: {slo_deadline:.2f}s "
              f"(q={slo_quantile}, cal_costs median={np.median(cal_costs):.2f}s, "
              f"max={np.max(cal_costs):.2f}s)")
        calibration[str(seed)] = {
            "slo_quantile": slo_quantile,
            "slo_deadline_sec": slo_deadline,
            "cal_costs_median_sec": float(np.median(cal_costs)),
            "cal_costs_max_sec": float(np.max(cal_costs)),
            "cal_costs_sec": {str(w): float(cc) for w, cc in zip(paired_clients, cal_costs)},
            "pairing": [[str(w), str(t)] for w, t in pairing],
        }

        seed_round_rows: list[dict] = []
        seed_summary_rows: list[dict] = []

        for policy in POLICIES:
            print(f"  -> Policy: {policy.upper()}")
            # All policies start from the exact identical warm_model
            model = copy.deepcopy(warm_model)
            rng = np.random.default_rng(seed + sum(map(ord, policy)))
            cost_preds = {w: cal_costs[i] for i, w in enumerate(paired_clients)}

            cur_clock = 0.0
            cum_bytes = 0
            misses = 0          # mid-round availability dropouts
            timeouts = 0        # clients cut by the server at the deadline
            fallback_rounds = 0 # rounds where FL-Control had to use fallback (Problem 8)
            round_times = []            # realized (server-side, capped at B) round latency
            round_times_uncapped = []   # straggler's would-be completion time
            slo_met_flags = []
            slo_latency_flags = []
            f1_history = []
            jain_history = []
            availability_counts = {w: 0 for w in paired_clients}
            selection_counts = {w: 0 for w in paired_clients}
            participation_counts = {w: 0 for w in paired_clients}
            time_to_quality = float("nan")
            reached_target = 0

            r = 1
            max_attempts = cfg.rounds * 10
            attempts = 0
            while r <= cfg.rounds and attempts < max_attempts:
                attempts += 1
                # 1. Check availability
                available = [w for w in paired_clients if paired_profiles[w].is_active(cur_clock)]
                if len(available) < cfg.clients_per_round:
                    # If not enough clients available right now, advance clock by 30s
                    cur_clock += 30.0
                    continue

                for w in available:
                    availability_counts[w] += 1

                # 2. Compute current client losses and sample volume
                losses = {w: local_loss(model, clients_pool[w]["train_X"], clients_pool[w]["train_y"]) for w in available}
                sample_sizes = {w: len(clients_pool[w]["train_y"]) for w in available}

                # 3. Select cohort
                selected, fallback_used = select_cohort(policy, available, losses, cost_preds,
                                                        slo_deadline, cfg.clients_per_round, rng,
                                                        sample_sizes=sample_sizes)
                fallback_rounds += fallback_used
                for w in selected:
                    selection_counts[w] += 1

                # 4. Execute training on selected clients
                states, weights = [], []
                round_client_times, uncapped_client_times = [], []
                round_miss = False
                round_timeout = False
                for j, w in enumerate(selected):
                    prof = paired_profiles[w]
                    c = clients_pool[w]

                    st, ep, _, _ = adaptive_local_train(
                        model, c["train_X"], c["train_y"],
                        seed=seed * 10000 + r * 100 + j,
                        max_epochs=10
                    )
                    actual_time = prof.estimate_round_time(len(c["train_y"]), ep, model_size)
                    uncapped_client_times.append(actual_time)

                    # The server waits for a client at most until the deadline B
                    wait_end = cur_clock + min(actual_time, slo_deadline)

                    # (a) Exact interval check: does the client drop out before finishing / before B?
                    drop_at = prof.first_inactive_time(cur_clock, wait_end)
                    if drop_at is not None:
                        round_client_times.append(max(drop_at - cur_clock, 0.0))
                        misses += 1
                        round_miss = True
                        continue  # no update received: excluded from aggregation and from feedback

                    # Measured-work feedback (Sec. III). A client cut by the deadline still
                    # reports its completion time later, so its prediction is also updated.
                    cost_preds[w] = 0.65 * cost_preds[w] + 0.35 * actual_time

                    # (b) Deadline cut: the server stops waiting at B
                    if actual_time > slo_deadline:
                        round_client_times.append(slo_deadline)
                        timeouts += 1
                        round_timeout = True
                        continue  # late update discarded

                    round_client_times.append(actual_time)
                    states.append(st)
                    weights.append(len(c["train_y"]))
                    participation_counts[w] += 1

                round_latency = max(round_client_times) if round_client_times else 0.0
                round_latency_uncapped = max(uncapped_client_times) if uncapped_client_times else 0.0
                cur_clock += round_latency
                round_times.append(round_latency)
                round_times_uncapped.append(round_latency_uncapped)

                # 5. Aggregate model
                if states:
                    model.load_state_dict(fedavg(states, weights))

                # 6. Global evaluation
                acc, f1 = evaluate_global(model, val_X, val_y)
                f1_history.append(f1)

                if reached_target == 0 and f1 >= target_f1:
                    time_to_quality = cur_clock
                    reached_target = 1

                # Per-round Jain fairness over local losses of all N clients
                round_losses = [local_loss(model, clients_pool[w]["train_X"], clients_pool[w]["train_y"])
                                for w in paired_clients]
                jains_round = jain_index(round_losses)
                jain_history.append(jains_round)

                # Latency-only SLO (original systems contract), kept as a secondary metric
                slo_latency_ok = int(not round_miss and not round_timeout)
                slo_latency_flags.append(slo_latency_ok)

                # Composite SLO: latency feasibility AND quality floor AND fairness floor (Problems 2 & 3)
                slo_satisfied = int(
                    slo_latency_ok
                    and f1 >= f1_min
                    and jains_round >= jain_min
                )
                slo_met_flags.append(slo_satisfied)
                cum_bytes += len(states) * model_size * 2

                seed_round_rows.append({
                    "workload": workload_name,
                    "seed": seed,
                    "policy": policy,
                    "round": r,
                    "f1": f1,
                    "accuracy": acc,
                    "jain_round": jains_round,
                    "round_time_sec": round_latency,
                    "round_time_uncapped_sec": round_latency_uncapped,
                    "slo_deadline_sec": slo_deadline,
                    "dropout": int(round_miss),
                    "timeout": int(round_timeout),
                    "fallback_used": fallback_used,
                    "n_aggregated": len(states),
                    "slo_latency_met": slo_latency_ok,
                    "f1_floor_met": int(f1 >= f1_min),
                    "jain_floor_met": int(jains_round >= jain_min),
                    "slo_met": slo_satisfied,
                    "wall_clock_sec": cur_clock,
                    "bytes_round": len(states) * model_size * 2
                })
                r += 1

            # Calculate summary metrics for this seed & policy
            p95_time = float(np.percentile(round_times, 95)) if round_times else 0.0
            p95_time_uncapped = float(np.percentile(round_times_uncapped, 95)) if round_times_uncapped else 0.0
            slo_rate = float(np.mean(slo_met_flags)) if slo_met_flags else 0.0
            slo_latency_rate = float(np.mean(slo_latency_flags)) if slo_latency_flags else 0.0
            mean_jain_round = float(np.mean(jain_history)) if jain_history else 0.0
            final_f1 = float(np.mean(f1_history[-5:])) if len(f1_history) >= 5 else 0.0

            # Jain's fairness index over final client training losses (loss uniformity)
            client_losses = [local_loss(model, clients_pool[w]["train_X"], clients_pool[w]["train_y"]) for w in paired_clients]
            jains_fairness = jain_index(client_losses)

            # Raw and availability-normalized fairness (Problem 6 Fix)
            jains_selection = jain_index(selection_counts[w] for w in paired_clients)
            jains_participation = jain_index(participation_counts[w] for w in paired_clients)
            selection_rates = [selection_counts[w] / max(availability_counts[w], 1) for w in paired_clients]
            participation_rates = [participation_counts[w] / max(availability_counts[w], 1) for w in paired_clients]
            jains_selection_norm = jain_index(selection_rates)
            jains_participation_norm = jain_index(participation_rates)

            seed_summary_rows.append({
                "workload": workload_name,
                "seed": seed,
                "policy": policy,
                "final_f1": final_f1,
                "p95_round_time": p95_time,
                "p95_round_time_uncapped": p95_time_uncapped,
                "slo_rate": slo_rate,
                "slo_latency_rate": slo_latency_rate,
                "mean_jain_round": mean_jain_round,
                "availability_misses": misses,
                "deadline_timeouts": timeouts,
                "flcontrol_fallback_rounds": fallback_rounds,
                "time_to_quality_sec": time_to_quality,   # NaN if target never reached
                "reached_target": reached_target,
                "fairness_index": jains_fairness,
                "fairness_selection": jains_selection,
                "fairness_selection_norm": jains_selection_norm,
                "fairness_participation": jains_participation,
                "fairness_participation_norm": jains_participation_norm,
                "total_bytes": cum_bytes,
                "total_wall_clock_sec": cur_clock,
                "slo_deadline_sec": slo_deadline,
                "rounds_completed": len(round_times)
            })

        # Checkpoint after every seed (survives Colab disconnections; use --resume to continue)
        _append_csv(seed_round_rows, rounds_ckpt)
        _append_csv(seed_summary_rows, seeds_ckpt)
        calib_path.write_text(json.dumps(calibration, indent=2))
        aggregate_seed_summaries(pd.read_csv(seeds_ckpt)).to_csv(agg_path, index=False)
        print(f"  [checkpoint] seed {seed} written to {seeds_ckpt.name}")

    summary_cfg = {
        "workload": workload_name,
        "seeds": seeds,
        "policies": POLICIES,
        "clients_per_round": cfg.clients_per_round,
        "rounds": cfg.rounds,
        "slo_rule": f"B = quantile(cal_costs, {slo_quantile}) per seed from {warmup_rounds}-round warm_model",
        "slo_quantile": slo_quantile,
        "warmup_rounds": warmup_rounds,
        "f1_min": f1_min,
        "jain_min": jain_min,
        "ttq_target_f1": target_f1,
        "out_suffix": out_suffix,
        "outputs_written": [str(rounds_ckpt), str(seeds_ckpt), str(agg_path), str(calib_path)]
    }
    config_path.write_text(json.dumps(summary_cfg, indent=2))
    print(f"\nExperiment complete! Aggregated results written to {agg_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", choices=WORKLOADS, required=True)
    ap.add_argument("--print-config", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Subset of seeds to run (default: all 20 SEEDS)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="Override the number of rounds of the workload")
    ap.add_argument("--slo-quantile", type=float, default=SLO_QUANTILE,
                    help="Quantile q of calibration costs used as deadline B (ablation: 0.50, 0.75, 0.90)")
    ap.add_argument("--f1-min", type=float, default=None,
                    help="F1 quality floor for composite SLO (default: 0.45 for wisdm2019, 0.20 for femnist)")
    ap.add_argument("--jain-min", type=float, default=0.50,
                    help="Jain fairness floor for composite SLO (default: 0.50)")
    ap.add_argument("--warmup-rounds", type=int, default=10,
                    help="Number of common FedAvg warmup rounds before calibration (default: 10)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Suffix for output files, e.g. _b1 or _q50")
    ap.add_argument("--resume", action="store_true",
                    help="Skip seeds already present in the checkpoint with the same suffix")
    args = ap.parse_args()
    if args.print_config:
        print(json.dumps(asdict(WORKLOADS[args.workload]), indent=2))
        return
    if not 0.0 < args.slo_quantile <= 1.0:
        ap.error("--slo-quantile must be in (0, 1]")
    run(args.workload, seeds=args.seeds, rounds=args.rounds,
        slo_quantile=args.slo_quantile, f1_min=args.f1_min, jain_min=args.jain_min,
        warmup_rounds=args.warmup_rounds, out_suffix=args.out_suffix, resume=args.resume)

if __name__ == "__main__":
    main()
