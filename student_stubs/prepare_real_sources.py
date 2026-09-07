#!/usr/bin/env python3
"""Print acquisition commands for the REAL student-extension datasets/traces."""
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
DST = PROJECT / 'data' / 'student_sources'
print(f'''STUDENT EXTENSION SOURCES

A) LEAF/FEMNIST natural writer clients
   git clone https://github.com/TalwalkarLab/leaf.git {DST/'leaf'}
   cd {DST/'leaf'/'data'/'femnist'}
   ./preprocess.sh -s niid --sf 1.0 -k 50 -t sample --tf 0.8 --smplseed 17 --spltseed 29

B) FedScale real environment traces
   git clone https://github.com/SymbioticLab/FedScale.git {DST/'FedScale'}
   Follow benchmark/dataset/README.md and obtain:
     benchmark/dataset/data/device_info/client_device_capacity
     benchmark/dataset/data/device_info/client_behave_trace

C) WISDM Smartphone/Smartwatch 2019
   Save the official UCI archive as:
   {DST/'WISDM_Smartphone_Smartwatch_2019.zip'}

Do not replace unavailable sources with generated CPU, bandwidth, latency,
availability, clients, or synthetic non-IID data.''')
