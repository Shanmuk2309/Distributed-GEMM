# 8192 MPI GEMM experiment

All 40 configurations completed and passed OpenBLAS verification: ten algorithm/precision variants at 1, 4, 9, and 16 MPI processes. The verified rows are appended to the corresponding per-program CSVs in `results/`.

The experiment ran sequentially on an Apple Silicon M5 MacBook Air with 16 GB memory, macOS 27, Open MPI 5.0.11, and Homebrew OpenBLAS 0.3.34. OpenBLAS and OpenMP thread counts were set to one. See `run_metadata.txt` for the run summary, `outcomes.csv` for each configuration, and `logs/` for complete program output.

These timings are from a different machine and MPI implementation than the original Windows campaign, so compare their correctness and behavior separately from the 512–4096 measurements.

To repeat the experiment on the same macOS toolchain, run `python3 scripts/run_8192_benchmarks.py` from the repository root.
