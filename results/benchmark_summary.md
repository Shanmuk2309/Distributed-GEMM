# MPI GEMM benchmark summary

- Configurations attempted in this run: 160
- Successful configurations in this run: 160
- Failed configurations in this run: 0
- Skipped configurations with an existing successful row: 0
- Measurements: one run per configuration; summary medians therefore equal each configuration's single successful measurement unless pre-existing successful records are present.
- Algorithms and precision variants: 1D GEMM, SUMMA, Cannon, SUMMA overlap, Cannon overlap; single and double precision.
- Matrix sizes: 512, 1024, 2048, 4096. Matrix size 8192 was not run.
- Process counts: 1, 4, 9, 16.
- MPI startup preflight: 9 and 16 processes passed with hello_mpi.exe; OpenBLAS-linked 512 GEMM passed at 9 and 16 processes.
- CSV files updated: successful rows appended to the mapped algorithm/precision CSVs; failures appended to benchmark_failures.csv when present.
- All MPI benchmark jobs were run sequentially.

## Successful configurations

| Program | Precision | Successful matrix/process configurations |
|---|---|---|
| oned_gemm | Single | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| oned_gemm_double | Double | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| summa | Single | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| summa_double | Double | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| cannon | Single | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| cannon_double | Double | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| summa_overlap | Single | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| summa_overlap_double | Double | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| cannon_overlap | Single | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |
| cannon_overlap_double | Double | 1024/P1, 1024/P16, 1024/P4, 1024/P9, 2048/P1, 2048/P16, 2048/P4, 2048/P9, 4096/P1, 4096/P16, 4096/P4, 4096/P9, 512/P1, 512/P16, 512/P4, 512/P9 |

## Failures

No failed benchmark configurations in this run.

- 9-process execution worked in preflight: yes.
- 16-process execution worked in preflight: yes.
- Incomplete configurations: see failed and skipped counts above; configurations not attempted due to an early preflight/compiler stop are not included.
