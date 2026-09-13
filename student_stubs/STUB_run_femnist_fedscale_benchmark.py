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
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
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

# Deterministic thread execution
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
SEEDS = [17, 29, 43, 59, 71, 89, 107, 131, 167, 197]
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
    "femnist": Workload("femnist", 3550, 20, 5, 100, "2-conv CNN + 128 FC, 62 classes"),
    "wisdm2019": Workload("wisdm2019", 51, 20, 5, 80, "1D-CNN 64/128 + global pooling, 18 classes"),
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
        self.behavior_index = 0

    def is_active(self, cur_time: float) -> bool:
        """Evaluate FLASH real client availability trace at given wall-clock time."""
        if not self.avail_trace:
            return True
        finish_time = self.avail_trace.get("finish_time", 86400 * 7)
        norm_time = cur_time % finish_time
        actives = self.avail_trace.get("active", [])
        inactives = self.avail_trace.get("inactive", [])
        if not actives or not inactives:
            return True

        # Advance behavior index to current segment
        while self.behavior_index < len(inactives) and norm_time > inactives[self.behavior_index]:
            self.behavior_index += 1
        self.behavior_index %= len(actives)

        return actives[self.behavior_index] <= norm_time <= inactives[self.behavior_index]

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
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        logits = model(xt)
        return float(F.cross_entropy(logits, yt).item())

def adaptive_local_train(global_model: nn.Module, X: np.ndarray, y: np.ndarray,
                         seed: int, rel_reduction=0.20, max_epochs=10, lr=0.01):
    model = copy.deepcopy(global_model)
    model.train()
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    n = len(yt)
    start_loss = local_loss(model, X, y)
    target_loss = start_loss * (1.0 - rel_reduction)
    epochs = 0

    for ep in range(1, max_epochs + 1):
        order = torch.randperm(n, generator=gen)
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
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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
        logits = model(torch.tensor(test_X, dtype=torch.float32))
        preds = logits.argmax(1).cpu().numpy()
    acc = accuracy_score(test_y, preds)
    f1 = f1_score(test_y, preds, average="macro", zero_division=0)
    return acc, f1

# ==========================================
# Client Selection Policies
# ==========================================
def select_cohort(policy: str, candidates: list[str], losses: dict[str, float],
                  cost_preds: dict[str, float], deadline: float, k: int,
                  rng: np.random.Generator, oort_alpha: float = 1.0):
    if policy == "random":
        return list(rng.choice(candidates, size=k, replace=False))

    if policy == "fastest":
        return sorted(candidates, key=lambda u: (cost_preds[u], u))[:k]

    if policy == "loss":
        return sorted(candidates, key=lambda u: (losses[u], -ord(str(u)[0])), reverse=True)[:k]

    if policy == "oort":
        # Oort utility: loss * min(1, (SLO / duration)^alpha)
        def oort_utility(u):
            duration = max(cost_preds[u], 1e-4)
            speed_penalty = min(1.0, (deadline / duration) ** oort_alpha)
            return losses[u] * speed_penalty

        return sorted(candidates, key=lambda u: oort_utility(u), reverse=True)[:k]

    if policy == "flcontrol":
        # Constrained optimization: max sum(losses) s.t. max(cost) <= deadline
        feasible = []
        for comb in itertools.combinations(candidates, k):
            max_c = max(cost_preds[u] for u in comb)
            if max_c <= deadline + 1e-9:
                feasible.append((sum(losses[u] for u in comb), -max_c, comb))
        if feasible:
            return list(max(feasible)[2])

        # If infeasible, minimize predicted violation then maximize learning utility
        cand = []
        for comb in itertools.combinations(candidates, k):
            max_c = max(cost_preds[u] for u in comb)
            cand.append((-max_c, sum(losses[u] for u in comb), comb))
        return list(max(cand)[2])

    raise ValueError(f"Unknown policy: {policy}")

# ==========================================
# Main Experiment Execution
# ==========================================
def run(workload_name: str):
    assert_sources(workload_name)
    cfg = WORKLOADS[workload_name]
    clients_pool = load_natural_workload_clients(workload_name)
    profiles_pool = load_fedscale_profiles()

    workload_ids = list(clients_pool.keys())
    trace_ids = list(profiles_pool.keys())

    all_round_metrics = []
    seed_summaries = []

    for seed in SEEDS:
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
        init_model = FemnistCNN(num_classes) if workload_name == "femnist" else Wisdm1DCNN(num_classes=num_classes)
        model_size = get_model_size_bytes(init_model)

        # Baseline calibration to determine the seed-specific SLO envelope
        cal_costs = []
        for w in paired_clients:
            c = clients_pool[w]
            st, ep, _, _ = adaptive_local_train(init_model, c["train_X"], c["train_y"], seed, max_epochs=10)
            est_t = paired_profiles[w].estimate_round_time(len(c["train_y"]), ep, model_size)
            cal_costs.append(est_t)

        # Fixed compute/system SLO deadline
        slo_deadline = float(np.quantile(cal_costs, 0.80))
        print(f"Seed {seed} SLO Deadline: {slo_deadline:.2f}s")

        for policy in POLICIES:
            print(f"  -> Policy: {policy.upper()}")
            model = copy.deepcopy(init_model)
            rng = np.random.default_rng(seed + sum(map(ord, policy)))
            cost_preds = {w: cal_costs[i] for i, w in enumerate(paired_clients)}

            cur_clock = 0.0
            cum_bytes = 0
            misses = 0
            round_times = []
            slo_met_flags = []
            f1_history = []
            fairness_losses = []

            for r in range(1, cfg.rounds + 1):
                # 1. Check availability
                available = [w for w in paired_clients if paired_profiles[w].is_active(cur_clock)]
                if len(available) < cfg.clients_per_round:
                    # Not enough available clients at this timestamp
                    cur_clock += 60.0  # Advance clock 1 minute
                    continue

                # 2. Compute current client losses
                losses = {w: local_loss(model, clients_pool[w]["train_X"], clients_pool[w]["train_y"]) for w in available}

                # 3. Select cohort
                selected = select_cohort(policy, available, losses, cost_preds,
                                         slo_deadline, cfg.clients_per_round, rng)

                # 4. Execute training on selected clients
                states, weights, round_client_times = [], [], []
                round_miss = False
                for j, w in enumerate(selected):
                    prof = paired_profiles[w]
                    c = clients_pool[w]

                    st, ep, _, _ = adaptive_local_train(
                        model, c["train_X"], c["train_y"],
                        seed=seed * 10000 + r * 100 + j,
                        max_epochs=10
                    )
                    actual_time = prof.estimate_round_time(len(c["train_y"]), ep, model_size)
                    round_client_times.append(actual_time)

                    # Verify client stayed available during execution
                    if not prof.is_active(cur_clock + actual_time):
                        misses += 1
                        round_miss = True
                    else:
                        states.append(st)
                        weights.append(len(c["train_y"]))

                    # Exponential moving average feedback update
                    cost_preds[w] = 0.65 * cost_preds[w] + 0.35 * actual_time

                round_latency = max(round_client_times) if round_client_times else 0.0
                cur_clock += round_latency
                round_times.append(round_latency)

                # 5. Aggregate model
                if states:
                    model.load_state_dict(fedavg(states, weights))

                # 6. Global evaluation
                acc, f1 = evaluate_global(model, val_X, val_y)
                f1_history.append(f1)
                slo_satisfied = int(round_latency <= slo_deadline and not round_miss)
                slo_met_flags.append(slo_satisfied)
                cum_bytes += len(states) * model_size * 2

                all_round_metrics.append({
                    "workload": workload_name,
                    "seed": seed,
                    "policy": policy,
                    "round": r,
                    "f1": f1,
                    "accuracy": acc,
                    "round_time_sec": round_latency,
                    "slo_deadline_sec": slo_deadline,
                    "slo_met": slo_satisfied,
                    "wall_clock_sec": cur_clock,
                    "bytes_round": len(states) * model_size * 2
                })

            # Calculate summary metrics for this seed & policy
            p95_time = float(np.percentile(round_times, 95)) if round_times else 0.0
            slo_rate = float(np.mean(slo_met_flags)) if slo_met_flags else 0.0
            final_f1 = float(np.mean(f1_history[-5:])) if len(f1_history) >= 5 else 0.0

            # Time to target quality (e.g. reaching 80% of final max F1)
            target_f1 = 0.8 * max(f1_history) if f1_history else 0.0
            time_to_quality = cur_clock
            for rm in [m for m in all_round_metrics if m["seed"] == seed and m["policy"] == policy]:
                if rm["f1"] >= target_f1:
                    time_to_quality = rm["wall_clock_sec"]
                    break

            # Jain's fairness index over individual client test losses
            client_losses = [local_loss(model, clients_pool[w]["train_X"], clients_pool[w]["train_y"]) for w in paired_clients]
            jains_fairness = float((sum(client_losses) ** 2) / (len(client_losses) * sum(l ** 2 for l in client_losses) + 1e-9))

            seed_summaries.append({
                "workload": workload_name,
                "seed": seed,
                "policy": policy,
                "final_f1": final_f1,
                "p95_round_time": p95_time,
                "slo_rate": slo_rate,
                "availability_misses": misses,
                "time_to_quality_sec": time_to_quality,
                "fairness_index": jains_fairness,
                "total_bytes": cum_bytes,
                "total_wall_clock_sec": cur_clock
            })

    # Save output artifacts
    df_rounds = pd.DataFrame(all_round_metrics)
    df_seeds = pd.DataFrame(seed_summaries)
    df_rounds.to_csv(OUT_DIR / f"round_metrics_{workload_name}.csv", index=False)
    df_seeds.to_csv(OUT_DIR / f"seed_summary_{workload_name}.csv", index=False)

    df_agg = df_seeds.groupby("policy").agg({
        "final_f1": ["mean", "std"],
        "p95_round_time": ["mean", "std"],
        "slo_rate": ["mean", "std"],
        "availability_misses": ["mean", "std"],
        "time_to_quality_sec": ["mean", "std"],
        "fairness_index": ["mean", "std"],
        "total_bytes": ["mean", "std"]
    }).reset_index()
    df_agg.to_csv(OUT_DIR / f"aggregate_{workload_name}.csv", index=False)

    summary_cfg = {
        "workload": workload_name,
        "seeds": SEEDS,
        "policies": POLICIES,
        "clients_per_round": cfg.clients_per_round,
        "rounds": cfg.rounds,
        "outputs_written": [
            str(OUT_DIR / f"round_metrics_{workload_name}.csv"),
            str(OUT_DIR / f"seed_summary_{workload_name}.csv"),
            str(OUT_DIR / f"aggregate_{workload_name}.csv")
        ]
    }
    (OUT_DIR / f"experiment_config_{workload_name}.json").write_text(json.dumps(summary_cfg, indent=2))
    print(f"\nExperiment complete! Aggregated results written to {OUT_DIR / f'aggregate_{workload_name}.csv'}")

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
