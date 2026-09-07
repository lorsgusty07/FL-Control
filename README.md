# Code layout

- `reference_run/` - executable experiment reproducing the paper figures/tables.
- `student_stubs/` - larger natural-client + real-trace experiments marked `TODO-STUDENT`.
- `requirements.txt` - shared dependencies.

Student extensions must use real sources and must not synthesize CPU, bandwidth, availability, or client partitions as fallback values.
