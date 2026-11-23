# Distributed GEMM with MPI and OpenBLAS

This project implements distributed matrix multiplication with MPI and verifies complete output matrices against OpenBLAS. It includes 1D row distribution, SUMMA, Cannon, and communication-overlap variants in single and double precision.

## Implementations

| Algorithm | Single precision | Double precision |
|---|---|---|
| 1D GEMM | `src/oned_gemm.c` | `src/oned_gemm_double.c` |
| SUMMA | `src/summa.c` | `src/summa_double.c` |
| Cannon | `src/cannon.c` | `src/cannon_double.c` |
| SUMMA overlap | `src/summa_overlap.c` | `src/summa_overlap_double.c` |
| Cannon overlap | `src/cannon_overlap.c` | `src/cannon_overlap_double.c` |

The programs take the matrix size as their command-line argument. The Windows benchmark campaign uses sizes 512, 1024, 2048, and 4096, with MPI process counts 1, 4, 9, and 16. The Windows machine skipped 8192 because available memory was below the estimated working set. A separate Mac run completed all 40 size-8192 algorithm/precision/process configurations; its metadata and outcomes are in `results/8192_experiment/`, with nonempty run logs alongside them.

## Build

Use MSYS2 UCRT64 `mpicc` and OpenBLAS. From the repository root, this PowerShell loop compiles all ten programs into root-level executables:

```powershell
$sources = @(
    'oned_gemm', 'oned_gemm_double',
    'summa', 'summa_double',
    'cannon', 'cannon_double',
    'summa_overlap', 'summa_overlap_double',
    'cannon_overlap', 'cannon_overlap_double'
)

foreach ($name in $sources) {
    mpicc -O2 "src\$name.c" `
        -IC:/msys64/ucrt64/include/openblas `
        -LC:/msys64/ucrt64/lib -lopenblas -lm `
        -o "$name.exe"
    if ($LASTEXITCODE -ne 0) { throw "Compilation failed: $name" }
}
```

## Run a program

For example, run double-precision SUMMA on a 1024 × 1024 matrix with four processes:

```powershell
$env:Path = "C:\msys64\ucrt64\bin;C:\Program Files\Microsoft MPI\Bin;" + $env:Path
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:GOTO_NUM_THREADS = '1'
$env:OMP_DYNAMIC = 'FALSE'
mpiexec -n 4 .\summa_double.exe 1024
```

Use the corresponding executable name for the other implementations. Benchmark output includes execution, communication, and computation times; GFLOP/s; `C[0][0]`; maximum absolute and relative errors; and verification status.

## Benchmark records

The ten per-implementation CSV files are in `results/`. Their schema is:

```text
Matrix_Size,Processes,Precision,Execution_Time,Communication_Time,Computation_Time,GFLOPS,C00,Max_Absolute_Error,Max_Relative_Error,Verification
```

The campaign runner is `scripts/run_benchmarks.ps1`; its logs are in `results/benchmark_logs/`. The ten per-program CSVs in `results/` contain the 16 Windows measurements plus four Mac 8192 measurements each. `results/benchmark_summary.csv` has 200 one-run rows. The 8192 metadata, outcomes, and logs are under `results/8192_experiment/`. The Mac used a different machine and MPI implementation, so keep its performance comparisons separate from the Windows results.

## Graphs

Generate the project graphs with Python, pandas, and matplotlib:

```powershell
python scripts\generate_graphs.py
```

PNG figures at 300 DPI are saved under `results/graphs/`; analysis and report content are in `results/documentation/`. Graph generation validates all 200 CSV records but plots only the 160 Windows configurations at sizes 512–4096. The Mac 8192 measurements are intentionally excluded from those graphs so results from different machines are not mixed. Each point is a single-run measurement.
