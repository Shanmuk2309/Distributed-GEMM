# Distributed GEMM performance analysis

Charts include all 200 single-run measurements: 512–4096 were recorded on Windows and 8192 on a separate Mac. The 8192 measurements are visibly identified as Mac data; do not compare their absolute time or GFLOP/s directly with Windows points. Speedup and efficiency use each algorithm/precision/size's own P=1 result, so the 8192 scaling curves are within the Mac campaign. Each configuration ran once; summary medians are single measurements, not repeated-run averages. All raw CSV rows passed verification with maximum relative error below `1e-4`.

## Best-performing implementation by size

Winner means the shortest execution time among algorithms at the same matrix size, precision, and process count. This keeps the comparison fair. The winners can differ with precision or process count.

| Matrix size | Precision | Processes | Fastest algorithm | Execution time (s) | GFLOP/s |
|---:|---|---:|---|---:|---:|
| 512 | Single | 1 | SUMMA | 0.028509 | 9.416 |
| 512 | Single | 4 | Cannon | 0.016417 | 16.35 |
| 512 | Single | 9 | SUMMA overlap | 0.015356 | 17.48 |
| 512 | Single | 16 | Cannon overlap | 0.02166 | 12.39 |
| 512 | Double | 1 | 1D GEMM | 0.065989 | 4.068 |
| 512 | Double | 4 | SUMMA overlap | 0.035227 | 7.62 |
| 512 | Double | 9 | SUMMA | 0.029317 | 9.156 |
| 512 | Double | 16 | Cannon overlap | 0.06553 | 4.096 |
| 1,024 | Single | 1 | Cannon | 0.190474 | 11.27 |
| 1,024 | Single | 4 | SUMMA overlap | 0.108268 | 19.83 |
| 1,024 | Single | 9 | SUMMA overlap | 0.081991 | 26.19 |
| 1,024 | Single | 16 | SUMMA | 0.152219 | 14.11 |
| 1,024 | Double | 1 | Cannon overlap | 0.584642 | 3.673 |
| 1,024 | Double | 4 | SUMMA overlap | 0.276888 | 7.756 |
| 1,024 | Double | 9 | SUMMA | 0.199563 | 10.76 |
| 1,024 | Double | 16 | SUMMA overlap | 0.223449 | 9.611 |
| 2,048 | Single | 1 | SUMMA | 3.53435 | 4.861 |
| 2,048 | Single | 4 | Cannon overlap | 1.29142 | 13.3 |
| 2,048 | Single | 9 | SUMMA | 0.94293 | 18.22 |
| 2,048 | Single | 16 | Cannon | 0.759465 | 22.62 |
| 2,048 | Double | 1 | SUMMA | 5.92459 | 2.9 |
| 2,048 | Double | 4 | Cannon overlap | 4.04936 | 4.243 |
| 2,048 | Double | 9 | Cannon overlap | 3.68285 | 4.665 |
| 2,048 | Double | 16 | SUMMA | 2.10039 | 8.179 |
| 4,096 | Single | 1 | SUMMA overlap | 24.011 | 5.724 |
| 4,096 | Single | 4 | Cannon | 15.3981 | 8.926 |
| 4,096 | Single | 9 | SUMMA overlap | 14.7872 | 9.294 |
| 4,096 | Single | 16 | SUMMA | 14.7028 | 9.348 |
| 4,096 | Double | 1 | Cannon overlap | 51.2231 | 2.683 |
| 4,096 | Double | 4 | SUMMA | 32.235 | 4.264 |
| 4,096 | Double | 9 | Cannon overlap | 30.0464 | 4.574 |
| 4,096 | Double | 16 | SUMMA overlap | 30.4046 | 4.52 |
| 8,192 | Single | 1 | 1D GEMM | 33.3637 | 32.96 |
| 8,192 | Single | 4 | 1D GEMM | 27.4873 | 40 |
| 8,192 | Single | 9 | SUMMA overlap | 24.7435 | 44.44 |
| 8,192 | Single | 16 | Cannon | 26.6293 | 41.29 |
| 8,192 | Double | 1 | 1D GEMM | 80.638 | 13.64 |
| 8,192 | Double | 4 | Cannon | 57.0197 | 19.28 |
| 8,192 | Double | 9 | Cannon | 53.2182 | 20.66 |
| 8,192 | Double | 16 | SUMMA | 56.3526 | 19.51 |

## Process count, speedup, and efficiency

Speedup is computed as `T1/Tp` within each algorithm, precision, and matrix size; parallel efficiency is `(speedup/processes) × 100`. The plotted P=1 speedup and efficiency are set to exactly 1 and 100%, respectively.

| Precision | Size | Algorithm | P4 speedup | P9 speedup | P16 speedup | P16 efficiency |
|---|---:|---|---:|---:|---:|---:|
| Single | 512 | 1D GEMM | 1.362× | 1.135× | 0.637× | 4.0% |
| Single | 512 | SUMMA | 1.344× | 1.841× | 0.428× | 2.7% |
| Single | 512 | Cannon | 2.065× | 1.992× | 1.516× | 9.5% |
| Single | 512 | SUMMA overlap | 2.030× | 2.713× | 0.953× | 6.0% |
| Single | 512 | Cannon overlap | 1.813× | 2.144× | 1.761× | 11.0% |
| Single | 1,024 | 1D GEMM | 1.560× | 1.017× | 0.872× | 5.4% |
| Single | 1,024 | SUMMA | 1.555× | 2.436× | 1.348× | 8.4% |
| Single | 1,024 | Cannon | 1.593× | 2.180× | 1.184× | 7.4% |
| Single | 1,024 | SUMMA overlap | 1.950× | 2.575× | 1.316× | 8.2% |
| Single | 1,024 | Cannon overlap | 1.854× | 2.463× | 0.908× | 5.7% |
| Single | 2,048 | 1D GEMM | 1.700× | 1.989× | 1.806× | 11.3% |
| Single | 2,048 | SUMMA | 2.456× | 3.748× | 4.444× | 27.8% |
| Single | 2,048 | Cannon | 3.760× | 4.989× | 6.812× | 42.6% |
| Single | 2,048 | SUMMA overlap | 2.691× | 3.485× | 4.438× | 27.7% |
| Single | 2,048 | Cannon overlap | 3.537× | 4.586× | 5.943× | 37.1% |
| Single | 4,096 | 1D GEMM | 1.501× | 1.568× | 1.456× | 9.1% |
| Single | 4,096 | SUMMA | 1.703× | 1.824× | 1.875× | 11.7% |
| Single | 4,096 | Cannon | 1.880× | 1.915× | 1.258× | 7.9% |
| Single | 4,096 | SUMMA overlap | 1.019× | 1.624× | 1.587× | 9.9% |
| Single | 4,096 | Cannon overlap | 1.694× | 1.836× | 1.805× | 11.3% |
| Single | 8,192 | 1D GEMM | 1.214× | 1.062× | 1.225× | 7.7% |
| Single | 8,192 | SUMMA | 1.245× | 1.255× | 1.200× | 7.5% |
| Single | 8,192 | Cannon | 1.483× | 1.437× | 1.553× | 9.7% |
| Single | 8,192 | SUMMA overlap | 1.523× | 1.743× | 1.587× | 9.9% |
| Single | 8,192 | Cannon overlap | 1.498× | 1.604× | 1.517× | 9.5% |
| Double | 512 | 1D GEMM | 1.630× | 1.253× | 0.272× | 1.7% |
| Double | 512 | SUMMA | 1.936× | 2.872× | 0.514× | 3.2% |
| Double | 512 | Cannon | 1.498× | 1.254× | 0.594× | 3.7% |
| Double | 512 | SUMMA overlap | 2.082× | 2.487× | 0.711× | 4.4% |
| Double | 512 | Cannon overlap | 1.774× | 1.585× | 1.132× | 7.1% |
| Double | 1,024 | 1D GEMM | 0.965× | 1.107× | 1.048× | 6.5% |
| Double | 1,024 | SUMMA | 2.024× | 3.012× | 2.593× | 16.2% |
| Double | 1,024 | Cannon | 2.544× | 3.007× | 3.227× | 20.2% |
| Double | 1,024 | SUMMA overlap | 2.727× | 3.300× | 3.379× | 21.1% |
| Double | 1,024 | Cannon overlap | 2.020× | 2.455× | 2.287× | 14.3% |
| Double | 2,048 | 1D GEMM | 1.466× | 1.579× | 1.546× | 9.7% |
| Double | 2,048 | SUMMA | 1.276× | 1.593× | 2.821× | 17.6% |
| Double | 2,048 | Cannon | 1.711× | 2.042× | 3.419× | 21.4% |
| Double | 2,048 | SUMMA overlap | 2.090× | 2.413× | 4.171× | 26.1% |
| Double | 2,048 | Cannon overlap | 2.117× | 2.327× | 3.949× | 24.7% |
| Double | 4,096 | 1D GEMM | 3.638× | 3.699× | 3.127× | 19.5% |
| Double | 4,096 | SUMMA | 2.855× | 3.015× | 2.893× | 18.1% |
| Double | 4,096 | Cannon | 1.497× | 1.679× | 1.664× | 10.4% |
| Double | 4,096 | SUMMA overlap | 1.509× | 1.654× | 1.780× | 11.1% |
| Double | 4,096 | Cannon overlap | 1.488× | 1.705× | 1.517× | 9.5% |
| Double | 8,192 | 1D GEMM | 1.298× | 1.262× | 0.515× | 3.2% |
| Double | 8,192 | SUMMA | 1.475× | 1.531× | 1.525× | 9.5% |
| Double | 8,192 | Cannon | 1.640× | 1.757× | 1.598× | 10.0% |
| Double | 8,192 | SUMMA overlap | 1.699× | 1.808× | 1.641× | 10.3% |
| Double | 8,192 | Cannon overlap | 1.675× | 1.720× | 1.590× | 9.9% |

## Communication overhead

Across the 200 configurations from two machines, communication time is greater than computation time in 23 cases. The median communication share of the measured communication-plus-computation components is 6.8% (range 0.1%–94.9%). This component share is descriptive; for overlap implementations communication and computation may occur concurrently, so their sum is not necessarily execution time.

- At 16 processes (single), median reported communication time is 49.2% of execution time.
- At 16 processes (double), median reported communication time is 40.9% of execution time.

## Communication-computation overlap

- SUMMA (single): overlap was faster in 10/20 matched size/process comparisons; median execution-time change was -0.14% (positive means faster with overlap).
- SUMMA (double): overlap was faster in 10/20 matched size/process comparisons; median execution-time change was -0.24% (positive means faster with overlap).
- Cannon (single): overlap was faster in 9/20 matched size/process comparisons; median execution-time change was -1.06% (positive means faster with overlap).
- Cannon (double): overlap was faster in 12/20 matched size/process comparisons; median execution-time change was +2.77% (positive means faster with overlap).
These are comparisons of measured wall times only. Single-run results do not establish that observed differences are repeatable or statistically significant.

## SUMMA versus Cannon

- Plain SUMMA was faster in 11/20 matched cases and Cannon in 9/20 (single). No single algorithm dominated every tested size/process combination.
- Plain SUMMA was faster in 14/20 matched cases and Cannon in 6/20 (double). No single algorithm dominated every tested size/process combination.
Use the winner table above to see the specific size and process-count conditions; rankings are workload- and precision-dependent.

## Single versus double precision

- 1D GEMM: the double/single execution-time ratio has a median of 2.08× (range 1.65×–5.75×) across matched configurations.
- SUMMA: the double/single execution-time ratio has a median of 2.21× (range 1.52×–3.94×) across matched configurations.
- Cannon: the double/single execution-time ratio has a median of 2.28× (range 1.39×–5.33×) across matched configurations.
- SUMMA overlap: the double/single execution-time ratio has a median of 2.24× (range 1.39×–3.69×) across matched configurations.
- Cannon overlap: the double/single execution-time ratio has a median of 2.18× (range 1.07×–3.70×) across matched configurations.

## Measurement limitations

- Every configuration has one measurement. The summary median therefore has no repeated samples behind it; system load, process placement, and timing noise can affect individual points.
- The 512–4096 campaign used one Windows machine/runtime; the 8192 campaign used a separate Mac and Open MPI. Their performance values are kept separate. Neither is a multi-node study.
- GFLOP/s uses each recorded execution time and the same `2n³` operation count. It is useful for within-campaign comparisons, but it does not separate the benchmark from system-level effects.
- The stacked communication/computation figures show the reported timing components. In overlap variants, these components can overlap, so they should not be interpreted as additive portions of total execution time.

## Inputs and generated figures

The ten `results/*_float.csv` and `results/*_double.csv` files were read without modification and validated for all 200 rows. Every graph includes the 8192 observations. Absolute performance graphs identify the machine boundary; speedups use matched P=1 baselines from each size's own campaign.
