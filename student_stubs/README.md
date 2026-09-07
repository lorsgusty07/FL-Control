# Student experiment stubs - FL-Control

These files are extension exercises for students and are intentionally outside the paper narrative. The reference scenario in `code/reference_run/` is executable; the stubs below define larger trace-backed scenarios to run later on suitable machines.

## Scenario FC-S1 - FEMNIST + FedScale
- Workload clients: 20 natural FEMNIST writers selected deterministically per seed.
- Cohort size: K = 5.
- Model: 2-convolution CNN + 128-unit fully connected layer.
- Systems state: real FedScale AIBench compute, MobiPerf bandwidth, and FLASH availability traces.
- Pair workload client IDs and trace profiles deterministically and keep the mapping fixed across policies.

## Scenario FC-S2 - WISDM 2019 + FedScale
- Workload clients: 20 natural WISDM subjects selected from the real subject pool.
- Cohort size: K = 5.
- Model: 1-D CNN 64/128 + global pooling.
- Use the same real FedScale trace family as FC-S1.

## Student task
1. Follow `prepare_real_sources.py` to obtain LEAF/FEMNIST and FedScale sources.
2. Complete functions marked `TODO-STUDENT` in `STUB_run_femnist_fedscale_benchmark.py`.
3. Reuse the reference FL-Control policy semantics, paired seeds, and statistics.
4. Write real outputs to `results/student_runs/`.
5. Never synthesize CPU, bandwidth, availability, client IDs, or non-IID partitions as a fallback.
