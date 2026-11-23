# Distributed Dense Matrix–Matrix Multiplication (GEMM) using MPI

## CS72 report content | Cluster A – Problem 3

**Platform:** MPI distributed-memory programming with OpenBLAS verification  
**Student(s):** [TO BE COMPLETED]  
**Template note:** `Multicore_Project_Report_Template.docx` was not present in the inspected checkout. This draft follows the section structure provided in the task; adjust heading numbering and layout to the college template.

### Evidence convention

This draft is based on all ten MPI C sources, the ten per-variant result CSVs, `results/benchmark_summary.csv`, graphs, scripts, README, and Mac experiment evidence under `results/8192_experiment/`. The Windows campaign contains 160 single-run rows for 512–4096; the separate Mac campaign contains 40 single-run rows for 8192. All 200 per-configuration records pass verification. Performance data across the two machines are kept separate because they used different hardware and MPI implementations. Unavailable team details are marked for completion.

# 1. Abstract

Dense matrix multiplication is a fundamental kernel in scientific computing, simulation, machine learning, and numerical linear algebra. This project computes the distributed-memory product \(C=AB\) using MPI and compares five conventional implementations: row-distributed 1D GEMM, SUMMA, Cannon’s algorithm, SUMMA with communication–computation overlap, and Cannon with communication–computation overlap. Each has single-precision `float` and double-precision `double` versions. The 1D method distributes A rows using `MPI_Scatterv`, broadcasts complete B, computes local C rows, and gathers with `MPI_Gatherv`. The 2D methods use Cartesian process grids and padded blocks; SUMMA broadcasts panels, while Cannon initially skews blocks and then shifts them periodically between neighboring ranks. The overlap versions use double buffers with nonblocking `MPI_Ibcast` or `MPI_Isend`/`MPI_Irecv` operations and completion waits.

The Windows campaign contains 160 single-run configurations at dimensions 512, 1024, 2048, and 4096 and process counts 1, 4, 9, and 16. A separate 8192 campaign ran all 40 algorithm/precision/process combinations on a friend’s Apple Silicon M5 MacBook Air with 16 GB RAM and Open MPI 5.0.11. Every result was compared with `cblas_sgemm` or `cblas_dgemm`; all 200 rows passed the maximum-relative-error threshold of `1e-4`. Across both machines the largest relative error was **7.2806769822e-06**, and the largest absolute error was **1.4770507812e-02**. The Mac data are a separate dataset and are not merged into the Windows graph comparisons.

The fastest algorithm varies with size, process count, and precision. At 4096, single precision was fastest with SUMMA at 16 processes; double precision was fastest with Cannon overlap at 9. Overlap did not improve every matched case. The contribution is a correctness-checked implementation and matched one-machine comparison of communication strategies. Since each configuration was measured once, the results are descriptive and do not establish statistical significance or universal algorithm superiority.

# 2. Introduction and Literature Survey

## 2.1 Background and Motivation

For square matrices \(A,B\in\mathbb{R}^{n\times n}\),

\[
C_{ij}=\sum_{k=0}^{n-1}A_{ik}B_{kj},\qquad 0\le i,j<n.
\]

The conventional triple-loop algorithm performs about \(n^3\) multiplications and \(n^3\) additions, or approximately \(2n^3\) floating-point operations, giving \(O(n^3)\) arithmetic complexity. Three dense matrices require \(3n^2\) scalar elements. Matrix products are central to scientific simulation, numerical linear algebra, transformations, optimization, and many machine-learning operations.

A serial execution assigns all arithmetic to one execution stream and one process’s memory. Parallel execution partitions matrix data and computation among processes. In distributed-memory systems, processes own private memory and exchange operands and results explicitly. A useful decomposition must balance local work against data movement and synchronization. Increasing process count can reduce local arithmetic but raise communication, startup, and scheduling overhead.

MPI standardizes process communicators, topologies, collectives, point-to-point operations, and completion semantics. This project uses `Scatterv`, `Bcast`, `Gatherv`, `Scatter`, `Gather`, `Reduce`, Cartesian topologies, blocking neighbor exchanges, and nonblocking operations. Nonblocking calls permit a schedule to overlap communication and computation, but do not alone guarantee asynchronous progress; MPI implementation behavior affects the outcome, as discussed in the literature below.

## 2.2 Literature Survey

| Ref. | Authors, title, publication, year | Main contribution and relevance |
|---|---|---|
| [1] | L. E. Cannon, *A Cellular Computer to Implement the Kalman Filter Algorithm*, Ph.D. thesis, Montana State University, 1969. [Thesis copy](https://scholarworks.montana.edu/server/api/core/bitstreams/643669d9-055d-4345-a467-41ad612599c8/content). | Early cellular/mesh computational architecture. Historical foundation for the periodic 2D multiplication algorithm implemented here. |
| [2] | H. Gupta and P. Sadayappan, “Communication Efficient Matrix Multiplication on Hypercubes,” *Proc. 6th ACM Symposium on Parallel Algorithms and Architectures (SPAA)*, 1994, pp. 320–329. [DOI: 10.1145/181014.181434](https://doi.org/10.1145/181014.181434). | Analyzes distributed dense multiplication communication patterns and topology. Context for balancing communication volume and computation. |
| [3] | R. A. van de Geijn and J. Watts, “SUMMA: Scalable Universal Matrix Multiplication Algorithm,” *Concurrency: Practice and Experience*, vol. 9, no. 4, pp. 255–274, 1997. [DOI](https://doi.org/10.1002/(SICI)1096-9128(199704)9:4%3C255::AID-CPE250%3E3.0.CO;2-2). | Presents a scalable panel-distribution approach to matrix multiplication; direct algorithmic reference for the SUMMA variants. |
| [4] | D. Irony, S. Toledo, and A. Tiskin, “Communication Lower Bounds for Distributed-Memory Matrix Multiplication,” *Journal of Parallel and Distributed Computing*, vol. 64, no. 9, pp. 1017–1026, 2004. [DOI: 10.1016/j.jpdc.2004.03.021](https://doi.org/10.1016/j.jpdc.2004.03.021). | Develops communication lower bounds for conventional distributed multiplication; gives theoretical context for 2D algorithms without predicting this machine’s timings. |
| [5] | MPI Forum, *MPI: A Message-Passing Interface Standard, Version 4.1*, 2023. [Official document](https://www.mpi-forum.org/docs/mpi-4.1/). | Defines communicator, topology, collective, nonblocking, and completion semantics used by these programs. |
| [6] | T. Hoefler, A. Lumsdaine, and W. Rehm, “Implementation and Performance Analysis of Non-Blocking Collective Operations for MPI,” *Proc. ACM/IEEE Conference on Supercomputing (SC ’07)*, 2007, pp. 1–10. [DOI: 10.1145/1362622.1362692](https://doi.org/10.1145/1362622.1362692). | Presents LibNBC and studies implementation/performance of nonblocking collectives; relevant to SUMMA overlap. |
| [7] | A. Denis, J. Jaeger, E. Jeannot, and F. Reynier, “A Methodology for Assessing Computation/Communication Overlap of MPI Nonblocking Collectives,” *Concurrency and Computation: Practice and Experience*, vol. 34, no. 22, e7168, 2022. [DOI: 10.1002/cpe.7168](https://doi.org/10.1002/cpe.7168). | Discusses MPI progress, overlap measurement, and cases where overlap helps, is absent, or hurts. It motivates cautious interpretation of this project’s timings. |
| [8] | L. S. Blackford et al., *ScaLAPACK Users’ Guide*. SIAM, 1997. [Netlib edition](https://www.netlib.org/scalapack/slug/). | Authoritative technical reference for process grids and distributed dense linear algebra data layouts. |

## 2.3 What Our Study Adds

This project implements five MPI strategies in two precisions. The 512–4096 comparison uses matched process counts on the Windows machine, and a separate 8192 campaign ran the same variants on a macOS M5 laptop. The work contrasts row decomposition, 2D panel broadcasts, periodic shifts, and explicit overlap schedules using a fixed seed convention and OpenBLAS validation. Each configuration was measured once; this is not a repeated-trial study, and performance data from the two machines must not be directly ranked.

# 3. Objectives, Problem Statement and Team Contribution

## 3.1 Problem Statement

Given dense square A and B of order n, compute C=AB on P MPI processes and compare different data distributions and communication patterns. Each method must produce the complete output and verify it against OpenBLAS. The experimental objective is to compare elapsed time and instrumented communication/computation components while changing algorithm, precision, size, and process count.

## 3.2 Objectives

1. Implement row-distributed 1D GEMM with uneven rows handled by `MPI_Scatterv`/`MPI_Gatherv`.
2. Implement SUMMA on a 2D Cartesian grid with row and column subcommunicators.
3. Implement Cannon with initial skewing and periodic neighbor exchanges.
4. Implement SUMMA and Cannon variants that schedule nonblocking communication alongside current-block computation.
5. Support float and double with matching MPI types and CBLAS reference routines.
6. Generate fixed-seed data and verify every output with maximum relative error below `1e-4`.
7. Measure maximum-rank execution, communication, computation, and GFLOP/s.
8. Analyze scaling, efficiency, accuracy, communication, precision, and overlap for the prescribed 160 configurations.

## 3.3 Team Contribution

| Name | USN | Implementation work | Experiments | Report sections/evidence |
|---|---|---|---|---|
| [TEAM MEMBER 1] | [USN] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| [TEAM MEMBER 2] | [USN] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| [Add rows as needed] | [USN] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |

# 4. Kernel, Algorithm and Dataset Description

## 4.1 Algorithm and Complexity

The operation is `C[i*n+j] += A[i*n+k] * B[k*n+j]`, accumulated over k. Classical GEMM performs \(O(n^3)\) work and approximately \(2n^3\) floating-point operations. A, B, C, and a full reference C together account for at least four n×n arrays at the verifying root; each array is \(n^2s\) bytes where s is 4 for float and 8 for double. Padded/packed arrays, local blocks, MPI runtime memory, and temporary verification copies add to the working set.

### 1D Distributed GEMM

`src/oned_gemm.c` and `src/oned_gemm_double.c` divide A’s rows evenly, giving the first n mod P ranks one additional row. Root builds element counts/displacements and calls `MPI_Scatterv`. Root broadcasts the complete B with `MPI_Bcast`, so each rank stores n² B elements. Each rank computes its local C rows using an i-k-j loop, then root reconstructs C with `MPI_Gatherv`. There is no reduction of partial C values because each rank owns distinct output rows.

### SUMMA

`src/summa.c` and `src/summa_double.c` make a q×q nonperiodic Cartesian process grid, q=√P, and create row/column communicators using `MPI_Cart_sub`. They set block edge b=ceil(n/q) and padded dimension N=bq. Root zero-initializes padded A and B, fills their original n×n portions, packs equal b×b blocks, and scatters them. At each of q stages, one A panel is broadcast across a process row and one B panel down a process column. Each rank performs its local block multiply and accumulates C. The result blocks are gathered, mapped back to original matrix positions, cropped to n×n for verification, and compared with OpenBLAS.

### Cannon’s algorithm

`src/cannon.c` and `src/cannon_double.c` use a periodic q×q Cartesian grid. Root packs and scatters padded equal-sized blocks. The initial skew shifts A left by each process-row coordinate and B upward by each process-column coordinate. Each of q stages multiplies the aligned local blocks; between stages, A shifts left and B shifts upward by one neighbor using `MPI_Sendrecv_replace`, with neighbors found through `MPI_Cart_shift`. Gathered blocks reconstruct C. The periodic topology wraps data around the grid.

### SUMMA overlap

`src/summa_overlap.c` and `src/summa_overlap_double.c` preserve the SUMMA process grid and panel decomposition and use current/next panel buffers. Initial panels are started with `MPI_Ibcast` and completed by `MPI_Waitall`. For later stages, next A/B broadcasts are initiated into inactive buffers before the current local product. After current-panel computation the ranks wait for outstanding broadcasts and swap buffer indices. This schedule exposes a communication-computation overlap window. Actual background progress is MPI-runtime dependent, so nonblocking operations do not guarantee hidden communication [5]–[7].

### Cannon overlap

`src/cannon_overlap.c` and `src/cannon_overlap_double.c` preserve Cannon’s periodic grid and initial blocking skew. Separate `nextA`/`nextB` receive buffers allow the program to post `MPI_Irecv` from the incoming neighbors and `MPI_Isend` current blocks to outgoing neighbors (A and B use distinct tags). Each rank computes using the current blocks, calls `MPI_Waitall`, and swaps current/next pointers before the next stage. The current buffers are read, not modified, before outstanding sends complete. This is actual communication scheduling during work; the amount hidden still depends on MPI progress and system contention.

### Overlap interpretation

Overlap can hide latency only if the MPI runtime progresses transfers during the local product. For small blocks, request setup/wait overhead can exceed hidden time. If progress requires MPI entry, transfers may wait until `MPI_Waitall`. Process scheduling and shared-memory contention can also negate gains. Therefore overlap is a hypothesis tested by matched elapsed-time pairs, not a guaranteed optimization.

## 4.2 Datasets and Inputs

All ten MPI programs seed the C library generator with `srand(42)` and generate elements using `rand()/RAND_MAX`, cast to float or double. The 1D source generates A and B values in an interleaved loop; 2D sources fill original matrix positions in padded row-major storage. The seed makes runs reproducible under the same C runtime; the C standard does not require `rand()` sequences to match across all platforms. Dense square sizes in the campaign are 512, 1024, 2048, and 4096. 2D variants pad dimensions to the next multiple of q and zero-initialize the padded regions.

## 4.3 Verification Method

Rank zero computes a complete OpenBLAS reference using `cblas_sgemm` or `cblas_dgemm`, then compares all n² entries. Absolute error is \(|C_{MPI}-C_{ref}|). Relative error divides by reference magnitude with a small denominator floor (the sources use approximately `1e-7` in 2D and `1e-12` in 1D). PASS requires maximum relative error below `1e-4`. Small differences are expected because floating-point summation order and optimized BLAS kernels differ. Verification occurs outside the distributed timed region and is a correctness check, not the performance baseline.

# 5. Experimental Setup and Hardware Characterisation

The campaign ran on one Windows machine using MSYS2 UCRT64 compiler tooling and Microsoft MPI launcher. CPU/OS/RAM CIM queries were unavailable in this inspection, so exact CPU model, core count, memory speed, and Windows build are not verified here. The supplied brief reports 8 GB RAM; confirm it on the machine before submission. Do not invent cache size, peak FLOP/s, or bandwidth.

| Component | Evidence |
|---|---|
| CPU model and core/logical counts | [TO BE COMPLETED; system query unavailable] |
| Installed RAM | 8 GB per supplied project notes; confirm locally |
| Memory speed | [TO BE COMPLETED; unconfirmed] |
| OS | Windows environment; exact Windows 11 release/build [TO BE COMPLETED] |
| Compiler | `C:\msys64\ucrt64\bin\mpicc.exe`; wrapper compiler reports GCC/cc 16.2.0, MSYS2 build |
| MPI runtime | Microsoft MPI launcher at `C:\Program Files\Microsoft MPI\Bin\mpiexec.exe`, file version 10.1.12498.52; `mpicc -show` links `libmsmpi.dll.a` |
| OpenBLAS | UCRT64 `openblas_config.h` identifies OpenBLAS 0.3.34 |
| Compile flags | `-O2 -lopenblas -lm` and UCRT64 OpenBLAS include/library paths |
| Thread settings | `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `GOTO_NUM_THREADS=1`, `OMP_DYNAMIC=FALSE` |
| Separate Mac experiment | Apple Silicon M5 MacBook Air, 16 GB RAM, macOS 27.0.1, Open MPI 5.0.11, Homebrew OpenBLAS 0.3.34 per supplied run metadata |
| Nodes/interconnect/GPU | Each campaign used one machine; no inter-node network or GPU characterization |

**8192 platform distinction.** The original Windows machine’s assessment reported 0.72 GiB available against a 1.50 GiB minimum estimated working set, so 8192 was not launched there. A separate friend’s Mac run bundle reports an Apple Silicon M5 MacBook Air, 16 GB RAM, macOS 27.0.1, Open MPI 5.0.11, and Homebrew OpenBLAS 0.3.34; that machine completed all 40 configurations. This is not a contradiction: Windows was skipped for resources, while the separate Mac run succeeded. Keep their performance data separate.

## 5.1 Measurement Methodology

Each program calls `MPI_Barrier` before the timed distributed phase and uses `MPI_Wtime`. The interval includes the distributed work/data movement and gather as implemented; rank-zero OpenBLAS verification is outside it. Each rank records total, communication, and computation intervals; three `MPI_Reduce` calls with `MPI_MAX` report the maximum for each metric. GFLOP/s is \(2n^3/(T_{exec}\cdot10^9)\), using original n even if the algorithm pads internally. Communication and computation maxima are reduced independently, so their sum need not equal maximum total time. In overlap code, communication and computation timing windows can overlap by design.

The process counts are 1, 4, 9, and 16. The runner limits OpenBLAS/OpenMP threading to one and executes one MPI job at a time. Nine- and sixteen-process preflight runs passed. On one machine, 9 and 16 ranks can oversubscribe CPUs depending on the unverified core count. Each configuration was measured once; there is no standard deviation or confidence interval.

# 6. Sequential Baseline and Verification

`src/serial_gemm.c` is a standalone naive float triple-loop at fixed size 512 using `clock()`. It is not an optimized `-O3` sequential baseline, does not cover all campaign sizes, and does not use the same MPI timing method. `src/test_blas.c` is a 2×2 CBLAS smoke program, not a GEMM timing baseline. Thus the inspected repository does not contain the template’s requested optimized sequential baseline.

Speedup below is defined relative to the same algorithm/precision/size’s P=1 MPI execution: \(S_p=T_{MPI,1}/T_{MPI,p}\); efficiency is \(100S_p/p\). It measures MPI scaling from one MPI rank, not speedup versus an optimized standalone sequential implementation. OpenBLAS verification is separate and its time is not included in the distributed execution time.

# 7. Parallel Implementation

## 7.1 Message-Passing Implementation (MPI)

1D uses `MPI_COMM_WORLD`, variable row counts, `Scatterv`, a full B broadcast, and `Gatherv`. SUMMA/Cannon variants use square Cartesian process grids for the tested square process counts. SUMMA creates row/column subcommunicators and broadcasts one A/B panel pair at each stage. Cannon uses a periodic grid, Cartesian neighbor lookup, initial skewing, and blocking shifts. SUMMA overlap uses `MPI_Ibcast` and alternating panel buffers. Cannon overlap uses tagged `MPI_Isend`/`MPI_Irecv`, distinct incoming buffers, local computation, `MPI_Waitall`, then swaps. All use a pre-timing barrier; allocation failures invoke `MPI_Abort`; 2D programs free their communicators before finalization.

## 7.4 Optimisation Steps

| Step | Choice | Expected benefit | Actual evidence; no universal performance claim |
|---|---|---|---|
| Row distribution | Scatter A rows and gather distinct C rows | Avoids partial-result reduction | `src/oned_gemm.c`, `src/oned_gemm_double.c`: Scatterv/Bcast/Gatherv; supports uneven row counts |
| 2D grids | Distribute both dimensions | Localizes panel communication to rows/columns | SUMMA/Cannon use q×q Cartesian grids; measured winners vary by case |
| Block/padding | Equal b×b messages | Simple fixed-count block collectives | `b=ceil(n/q)`, `N=bq`; padded inputs are zero-filled |
| Cannon skew | Align first operand blocks | Enables matching local products at each stage | Coordinate-dependent initial shifts in `src/cannon*.c` |
| Nonblocking transfers | Start next data movement before current work ends | Creates potential overlap window | `MPI_Ibcast` or `MPI_Isend`/`MPI_Irecv` followed by compute and `MPI_Waitall` |
| Double buffering | Keep current data separate from in-flight next data | Prevents reuse before communication completes | Panel indices or next-block pointers swap only after waits |
| Thread control | Limit BLAS/OpenMP threads | Avoids rank-wise thread oversubscription in verification/runtime | Runner sets OpenBLAS, OMP, GOTO to 1 and OMP_DYNAMIC FALSE |

## 7.5 Key Code Excerpts

**1D distribution** (`src/oned_gemm.c`; double uses `MPI_DOUBLE`):

```c
MPI_Scatterv(A, sendcounts, displs, MPI_FLOAT,
             local_A, local_rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
MPI_Bcast(B, (int)total_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);
/* local i-k-j product */
MPI_Gatherv(local_C, local_rows * n, MPI_FLOAT,
            C, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
```

Uneven row counts are represented by per-rank counts and displacements. B is replicated on every rank.

**SUMMA panels** (`src/summa.c`):

```c
MPI_Bcast(panelA, (int)block_size, MPI_FLOAT, k, row_comm);
MPI_Bcast(panelB, (int)block_size, MPI_FLOAT, k, col_comm);
```

The row/column subcommunicators distribute the panels needed for each local block product.

**Cannon topology and shift** (`src/cannon.c`):

```c
MPI_Cart_shift(grid, 1, -1, &right, &left);
MPI_Cart_shift(grid, 0, -1, &down, &up);
MPI_Sendrecv_replace(localA, (int)bs, MPI_FLOAT, left, 0,
                     right, 0, grid, MPI_STATUS_IGNORE);
MPI_Sendrecv_replace(localB, (int)bs, MPI_FLOAT, up, 1,
                     down, 1, grid, MPI_STATUS_IGNORE);
```

Initial shifts are repeated by process coordinates; after each stage, blocks shift one position on the periodic grid.

**SUMMA overlap** (`src/summa_overlap.c`):

```c
MPI_Ibcast(panelA[next], (int)bs, MPI_FLOAT, k + 1, row_comm, &req[0]);
MPI_Ibcast(panelB[next], (int)bs, MPI_FLOAT, k + 1, col_comm, &req[1]);
/* compute with panelA[current] and panelB[current] */
MPI_Waitall(2, req, statuses);
current = next;
```

Current and next panels are distinct until broadcast completion.

**Cannon overlap** (`src/cannon_overlap.c`):

```c
MPI_Irecv(nextA, (int)bs, MPI_FLOAT, right, 2, grid, &req[count++]);
MPI_Isend(localA, (int)bs, MPI_FLOAT, left, 2, grid, &req[count++]);
MPI_Irecv(nextB, (int)bs, MPI_FLOAT, down, 3, grid, &req[count++]);
MPI_Isend(localB, (int)bs, MPI_FLOAT, up, 3, grid, &req[count++]);
/* compute with current blocks; then MPI_Waitall and swap buffers */
```

A and B use different tags and receive buffers; the double variant substitutes `MPI_DOUBLE`.

# 8. Experiments, Results and Graphs

## 8.1 Benchmark configuration and observed winners

The ten raw CSVs in `results/` contain 200 rows total: 160 Windows records plus 40 Mac records. All pass OpenBLAS verification. The fastest row in Table 8.1 is selected only within a matched Windows size/precision/process group.

### Table 8.1. Fastest algorithm per exact configuration

| n | Precision | P | Fastest algorithm | Execution (s) | GFLOP/s |
|---:|---|---:|---|---:|---:|
| 512 | Single | 1 | SUMMA | 0.028509 | 9.4159 |
| 512 | Single | 4 | Cannon | 0.016417 | 16.3510 |
| 512 | Single | 9 | SUMMA overlap | 0.015356 | 17.4810 |
| 512 | Single | 16 | Cannon overlap | 0.021660 | 12.3930 |
| 512 | Double | 1 | 1D GEMM | 0.065989 | 4.0679 |
| 512 | Double | 4 | SUMMA overlap | 0.035227 | 7.6203 |
| 512 | Double | 9 | SUMMA | 0.029317 | 9.1562 |
| 512 | Double | 16 | Cannon overlap | 0.065530 | 4.0963 |
| 1024 | Single | 1 | Cannon | 0.190474 | 11.2740 |
| 1024 | Single | 4 | SUMMA overlap | 0.108268 | 19.8350 |
| 1024 | Single | 9 | SUMMA overlap | 0.081991 | 26.1920 |
| 1024 | Single | 16 | SUMMA | 0.152219 | 14.1080 |
| 1024 | Double | 1 | Cannon overlap | 0.584642 | 3.6732 |
| 1024 | Double | 4 | SUMMA overlap | 0.276888 | 7.7558 |
| 1024 | Double | 9 | SUMMA | 0.199563 | 10.7610 |
| 1024 | Double | 16 | SUMMA overlap | 0.223449 | 9.6106 |
| 2048 | Single | 1 | SUMMA | 3.534350 | 4.8608 |
| 2048 | Single | 4 | Cannon overlap | 1.291420 | 13.3030 |
| 2048 | Single | 9 | SUMMA | 0.942930 | 18.2200 |
| 2048 | Single | 16 | Cannon | 0.759465 | 22.6210 |
| 2048 | Double | 1 | SUMMA | 5.924590 | 2.8998 |
| 2048 | Double | 4 | Cannon overlap | 4.049360 | 4.2426 |
| 2048 | Double | 9 | Cannon overlap | 3.682850 | 4.6648 |
| 2048 | Double | 16 | SUMMA | 2.100390 | 8.1794 |
| 4096 | Single | 1 | SUMMA overlap | 24.011000 | 5.7240 |
| 4096 | Single | 4 | Cannon | 15.398100 | 8.9257 |
| 4096 | Single | 9 | SUMMA overlap | 14.787200 | 9.2944 |
| 4096 | Single | 16 | SUMMA | 14.702800 | 9.3478 |
| 4096 | Double | 1 | Cannon overlap | 51.223100 | 2.6831 |
| 4096 | Double | 4 | SUMMA | 32.235000 | 4.2637 |
| 4096 | Double | 9 | Cannon overlap | 30.046400 | 4.5742 |
| 4096 | Double | 16 | SUMMA overlap | 30.404600 | 4.5203 |

### Table 8.2. Minimum execution time across algorithms and P for each size/precision

| n | Precision | P | Winner | Execution (s) | GFLOP/s |
|---:|---|---:|---|---:|---:|
| 512 | Single | 9 | SUMMA overlap | 0.015356 | 17.4810 |
| 512 | Double | 9 | SUMMA | 0.029317 | 9.1562 |
| 1024 | Single | 9 | SUMMA overlap | 0.081991 | 26.1920 |
| 1024 | Double | 9 | SUMMA | 0.199563 | 10.7610 |
| 2048 | Single | 16 | Cannon | 0.759465 | 22.6210 |
| 2048 | Double | 16 | SUMMA | 2.100390 | 8.1794 |
| 4096 | Single | 16 | SUMMA | 14.702800 | 9.3478 |
| 4096 | Double | 9 | Cannon overlap | 30.046400 | 4.5742 |

### Table 8.3. Descriptive metric medians per algorithm and precision

Each median pools one observation at each of 16 size/process settings; it is descriptive only and should not be used as a same-workload ranking.

| Algorithm | Precision | Median exec (s) | Median comm (s) | Median comp (s) | Median GFLOP/s | Max relative error |
|---|---|---:|---:|---:|---:|---:|
| 1D GEMM | Double | 2.372094 | 0.129130 | 2.220949 | 3.8790 | 8.211e-15 |
| 1D GEMM | Single | 1.105861 | 0.064299 | 1.051500 | 8.1250 | 4.511e-06 |
| Cannon | Double | 1.524020 | 0.157554 | 1.270987 | 4.3299 | 8.659e-15 |
| Cannon | Single | 0.474969 | 0.087847 | 0.342644 | 12.2462 | 4.733e-06 |
| Cannon overlap | Double | 1.377472 | 0.148989 | 1.073508 | 4.4084 | 8.659e-15 |
| Cannon overlap | Single | 0.503964 | 0.094674 | 0.342862 | 11.1402 | 4.733e-06 |
| SUMMA | Double | 1.350720 | 0.159205 | 1.038893 | 4.4112 | 8.211e-15 |
| SUMMA | Single | 0.500220 | 0.090610 | 0.349692 | 11.2037 | 4.511e-06 |
| SUMMA overlap | Double | 1.473511 | 0.140026 | 1.169866 | 4.3605 | 8.211e-15 |
| SUMMA overlap | Single | 0.508105 | 0.077581 | 0.341493 | 11.5544 | 4.511e-06 |

### Table 8.4. Relative speedup and efficiency

For each n, speedup is same algorithm/precision’s P=1 MPI time divided by P time; this table shows the median across the four n values. Efficiency is speedup/P×100. P=1 is 1× and 100% by definition.

| Algorithm | Precision | P4 S | P4 E | P9 S | P9 E | P16 S | P16 E |
|---|---|---:|---:|---:|---:|---:|---:|
| 1D GEMM | Double | 1.548× | 38.7% | 1.416× | 15.7% | 1.297× | 8.1% |
| 1D GEMM | Single | 1.530× | 38.3% | 1.352× | 15.0% | 1.164× | 7.3% |
| Cannon | Double | 1.604× | 40.1% | 1.860× | 20.7% | 2.445× | 15.3% |
| Cannon | Single | 1.973× | 49.3% | 2.086× | 23.2% | 1.387× | 8.7% |
| Cannon overlap | Double | 1.897× | 47.4% | 2.016× | 22.4% | 1.902× | 11.9% |
| Cannon overlap | Single | 1.833× | 45.8% | 2.304× | 25.6% | 1.783× | 11.1% |
| SUMMA | Double | 1.980× | 49.5% | 2.942× | 32.7% | 2.707× | 16.9% |
| SUMMA | Single | 1.629× | 40.7% | 2.139× | 23.8% | 1.611× | 10.1% |
| SUMMA overlap | Double | 2.086× | 52.2% | 2.450× | 27.2% | 2.580× | 16.1% |
| SUMMA overlap | Single | 1.990× | 49.8% | 2.644× | 29.4% | 1.452× | 9.1% |

### Table 8.5. Verification maxima

| Algorithm | Precision | Maximum absolute error | Maximum relative error | Passing rows |
|---|---|---:|---:|---:|
| 1D GEMM | Single | 5.4931640625e-03 | 4.510706e-06 | 16/16 |
| 1D GEMM | Double | 1.00044417195e-11 | 8.210787e-15 | 16/16 |
| SUMMA | Single | 5.4931640625e-03 | 4.510706e-06 | 16/16 |
| SUMMA | Double | 1.00044417195e-11 | 8.210787e-15 | 16/16 |
| Cannon | Single | 5.4931640625e-03 | 4.733370e-06 | 16/16 |
| Cannon | Double | 1.023182e-11 | 8.658865e-15 | 16/16 |
| SUMMA overlap | Single | 5.4931640625e-03 | 4.510706e-06 | 16/16 |
| SUMMA overlap | Double | 1.00044417195e-11 | 8.210787e-15 | 16/16 |
| Cannon overlap | Single | 5.4931640625e-03 | 4.733370e-06 | 16/16 |
| Cannon overlap | Double | 1.023182e-11 | 8.658865e-15 | 16/16 |

### Figures and interpretations

All referenced files below are existing 300-DPI PNGs under `results/graphs/`. They were generated from `benchmark_summary.csv`; each summary median is based on one run per configuration, not repeated-run averages.

**Figure 8.1. Execution time versus matrix size, single precision.** `execution_time_single.png`. Curves compare algorithms at fixed process count. Larger n increases elapsed time, but the slope and ordering depend on algorithm and P. **Takeaway:** increasing P does not guarantee lower time at each workload.

**Figure 8.2. Execution time versus matrix size, double precision.** `execution_time_double.png`. Compare only matched n/P. **Takeaway:** double timings were usually larger, but the ratio is configuration-dependent.

**Figure 8.3–8.4. Speedup versus process count, single and double.** `speedup_single.png`, `speedup_double.png`. Each curve uses its same algorithm/precision/n P=1 MPI time. **Takeaway:** gains are sublinear and can reverse at small n or higher P.

**Figure 8.5–8.6. Parallel efficiency versus process count, single and double.** `parallel_efficiency_single.png`, `parallel_efficiency_double.png`. Efficiency is speedup/P×100, with P=1 fixed at 100%. **Takeaway:** efficiency decreases with additional ranks in many workloads.

**Figure 8.7–8.8. GFLOP/s versus matrix size, single and double.** `gflops_single.png`, `gflops_double.png`. Throughput uses 2n³ divided by elapsed time. **Takeaway:** measured throughput varies with size and P and is not a theoretical peak measurement.

**Figures 8.9–8.16. Computation and communication components by precision and process count.** Each chart is a separate figure; all use stacked components. The bars are useful for comparing reported communication/computation at matching P and precision. The component maxima are independently reduced, and overlap intervals are concurrent, so stack height need not equal total execution time.

| Figure | Image | Explanation and takeaway |
|---|---|---|
| 8.9 | `computation_communication_single_p1.png` | Single precision, P=1; shows baseline local/communication components. The setup and result movement remain in the communication measurement. |
| 8.10 | `computation_communication_single_p4.png` | Single precision, P=4; compare algorithms at the same process count. Component stacks are not a strict partition of elapsed time. |
| 8.11 | `computation_communication_single_p9.png` | Single precision, P=9; reports component timing after 2D padding/grid distribution where applicable. |
| 8.12 | `computation_communication_single_p16.png` | Single precision, P=16; communication share can be substantial, but the plot alone does not prove a hardware bottleneck. |
| 8.13 | `computation_communication_double_p1.png` | Double precision, P=1; compare only with the matching single-precision size/algorithm. |
| 8.14 | `computation_communication_double_p4.png` | Double precision, P=4; communication and computation vary by algorithm and size. |
| 8.15 | `computation_communication_double_p9.png` | Double precision, P=9; charts show instrumented components, not network-only latency. |
| 8.16 | `computation_communication_double_p16.png` | Double precision, P=16; independently reduced and concurrent components need not sum to total time. |

**Figures 8.17–8.20. Blocking versus overlap variants.** Matched n/P pairs are compared within each figure. Existing CSV-derived analysis says SUMMA overlap is faster in 8/16 pairs in both precisions, with median changes of −0.14% single and +1.51% double (positive means faster); thus single’s median is a 0.14% slowdown. Cannon overlap is faster in 8/16 single and 12/16 double pairs, with median changes −0.41% and +4.17%. **Takeaway:** overlap effects are mixed, and one run cannot establish repeatability.

| Figure | Image | Caption and one-sentence takeaway |
|---|---|---|
| 8.17 | `summa_overlap_comparison_single.png` | SUMMA versus SUMMA overlap, single precision; eight of sixteen pairs were faster with overlap, but the median change indicates a slight slowdown. |
| 8.18 | `summa_overlap_comparison_double.png` | SUMMA versus SUMMA overlap, double precision; eight of sixteen pairs were faster and the median measured improvement was 1.51%. |
| 8.19 | `cannon_overlap_comparison_single.png` | Cannon versus Cannon overlap, single precision; eight of sixteen pairs were faster and the median change was a slight slowdown. |
| 8.20 | `cannon_overlap_comparison_double.png` | Cannon versus Cannon overlap, double precision; twelve of sixteen pairs were faster, with 4.17% median measured improvement. |

**Figure 8.21. Single- versus double-precision execution-time comparison.** No combined precision graph exists in `results/graphs/`; use `execution_time_single.png` and `execution_time_double.png` side by side or produce a matched-configuration plot from CSV. Do not combine unmatched n/P series. Existing analysis gives median double/single timing ratios of 2.01× (1D), 2.31× (SUMMA), 2.79× (Cannon), 2.31× (SUMMA overlap), and 2.33× (Cannon overlap). **Takeaway:** double was slower in the median matched comparison, with ratio dependent on workload.

**Sign audit before final submission:** the reported overlap percentages and case counts are copied from `results/documentation/graph_analysis.md`. The single SUMMA and Cannon overlap medians are negative under the report’s stated positive=faster convention. Recalculate paired changes from the summary CSV if changing the phrasing or sign; do not silently invert it.

## 8.2 Summary of Metrics

Tables 8.1–8.5 cover Windows execution/GFLOP/s winners, scaling-relative speedup and efficiency, descriptive communication/computation medians, and per-variant numerical error maxima. The Mac 8192 records are separate rows in the CSVs under `results/` and are excluded from the Windows comparison. Table 8.3 pools multiple n/P conditions and is not an apples-to-apples algorithm comparison.

## 8.3 Separate 8192 Mac experiment

All ten 8192 algorithm/precision variants ran sequentially at P=1,4,9,16 on the Mac identified in `results/8192_experiment/run_metadata.txt`. The outcomes CSV records 40 attempted, 40 successful, zero failed, and zero skipped runs. Each per-program CSV contains 20 rows: 16 Windows measurements plus four Mac 8192 measurements. All 40 Mac results are PASS. Their maximum relative errors are `7.2806769822e-06` (single) and `1.2467699631e-14` (double); maximum absolute errors are `1.4770507812e-02` and `2.5920599001e-11`. Across the 20 configurations per precision, single execution times range 24.743506–43.138946 s and double times range 53.218212–156.603870 s. These ranges summarize distinct configurations, not repeats. The 20 PNGs plot only the Windows 512–4096 campaign; they do not include 8192. `results/benchmark_summary.csv` contains the 160 Windows entries plus 40 one-run Mac 8192 entries.

# 9. Analysis and Discussion

### 9.1 Matrix size and runtime

Runtime rises substantially with n because conventional work grows cubically. The growth is not a pure cubic ratio in wall time: block padding, communication, local loop behavior, data movement, synchronization, and scheduling also affect each point. 8192 elapsed times exist only for the separate Mac campaign and must not be appended to Windows performance curves as if machine conditions were identical.

### 9.2 Process count and 1D scaling

The 1D approach distributes rows but broadcasts all n² elements of B to each rank. It is simple and incurs no partial-C reduction, but B replication and broadcast remain even as local row work shrinks. At small n, collective overhead and scheduling can dominate. The recorded efficiency plots show low P16 efficiency in many cases; exact core count is unavailable, so possible oversubscription is a plausible condition, not a proven cause.

### 9.3 SUMMA and Cannon

SUMMA distributes both operands and broadcasts panels within process rows and columns. Cannon also distributes blocks but uses periodic nearest-neighbor shifts after a skew. These layouts reduce full-matrix B replication and provide structured communication. The measured winner varies: the graph analysis finds SUMMA faster than Cannon in 8/16 matched single-precision comparisons and 12/16 double-precision comparisons. No algorithm dominates every size/process case. These results describe this implementation/runtime/machine only.

### 9.4 Communication and synchronization

The existing analysis reports communication time greater than computation time in 23 of 160 configurations and a median communication share of 16.5% of the reported communication-plus-computation components (range 0.1–94.9%). For P=16 it reports median communication at 64.5% of execution in single precision and 44.0% in double. These statistics pool only the tested configurations and one run each. Distribution, panel/shift operations, and gather all contribute as instrumented. Independently reduced maxima need not add to maximum total time; overlap timings can represent concurrent windows.

### 9.5 Overlap

The source implements actual schedules that initiate next data movement before the current product and use separate buffers until completion. Existing analysis reports SUMMA overlap faster in 8/16 pairs per precision and Cannon overlap faster in 8/16 single and 12/16 double pairs. Median changes are small or mixed. Nonblocking calls may not progress asynchronously, and posting/waiting can add cost. Repeated trials and dedicated progress measurements are needed before concluding overlap is reliably beneficial.

### 9.6 Precision and numerical accuracy

The matched median double/single execution ratios in the existing analysis range from 2.01× to 2.79× by algorithm. This is measured here, not a universal precision rule; implementation and system effects also contribute. All 160 outputs passed. The maximum relative error (4.7334e-06) is below the threshold by over a factor of 20. Float maximum absolute error is greater than double, while relative error is the threshold metric used for PASS.

### 9.7 Limitations

Both campaigns are single-machine MPI runs, not a multi-node interconnect study. The Windows machine has 8 GB RAM according to task notes; CPU and memory-speed details could not be independently queried. Its 8192 run was skipped because available memory was below the estimated working set. The Mac metadata reports 16 GB RAM and a completed separate 8192 campaign. Each point has one run, so there is no standard deviation/error bar. No optimized standalone sequential baseline, Roofline analysis, STREAM bandwidth, or hardware-counter profile was found. Do not claim memory-bound or compute-bound behavior. Higher process counts may oversubscribe CPU resources; the Windows core count was not verified.

# 10. Cluster Integration and Contribution to the Joint Paper

| Item | Entry |
|---|---|
| Cluster/problem | Cluster A – Problem 3 |
| Team/USNs | [TO BE COMPLETED] |
| Inputs/precision | Dense square matrices; seed convention 42; float and double |
| Verification tolerance | Max relative error < `1e-4` against OpenBLAS CBLAS |
| Tested matrix sizes/process counts | 512,1024,2048,4096; P=1,4,9,16 |
| Results | Ten `results/*_float.csv` and `results/*_double.csv` files |
| Joint-paper contribution | [TO BE COMPLETED after team coordination] |
| Other clusters’ shared inputs/results | [TO BE COMPLETED; not provided in inspected checkout] |
| Open issues | Attach report template if required; repeat runs and hardware characterization |

# 11. Conclusion and Future Work

The project implements five MPI GEMM approaches in single and double precision and records 160 Windows measurements for 512–4096 plus 40 separate Mac measurements for 8192. All passed full-matrix OpenBLAS verification with relative error below `1e-4`; the largest observed relative error across both campaigns was `7.2806769822e-06`. For the Windows comparisons, single-precision 4096 was fastest with SUMMA/P16 and double-precision 4096 with Cannon overlap/P9. The Mac 8192 values must be analyzed separately rather than ranked against Windows data. Overlap helps in some matched one-run comparisons but not consistently; none of the rankings is a repeated-trial conclusion.

The Windows 8192 benchmark was skipped before launch because the assessment gives 0.72 GiB available versus a 1.50 GiB minimum working-set estimate; a separate Mac with 16 GB RAM completed all 40 verified 8192 configurations. Future work should repeat measurements on each platform, add an explicitly optimized standalone sequential baseline, characterize CPU/affinity/memory/MPI configuration, run on multiple nodes, compare equivalent machines for scaling, test MPI+OpenMP, measure asynchronous progress directly, and collect Roofline/STREAM or hardware-counter evidence.

# 12. References

[1] L. E. Cannon, *A Cellular Computer to Implement the Kalman Filter Algorithm*, Ph.D. thesis, Montana State University, 1969. [Online thesis](https://scholarworks.montana.edu/server/api/core/bitstreams/643669d9-055d-4345-a467-41ad612599c8/content).

[2] H. Gupta and P. Sadayappan, “Communication Efficient Matrix Multiplication on Hypercubes,” in *Proc. 6th ACM Symposium on Parallel Algorithms and Architectures (SPAA)*, 1994, pp. 320–329, doi: [10.1145/181014.181434](https://doi.org/10.1145/181014.181434).

[3] R. A. van de Geijn and J. Watts, “SUMMA: Scalable Universal Matrix Multiplication Algorithm,” *Concurrency: Practice and Experience*, vol. 9, no. 4, pp. 255–274, 1997, doi: [10.1002/(SICI)1096-9128(199704)9:4%3C255::AID-CPE250%3E3.0.CO;2-2](https://doi.org/10.1002/(SICI)1096-9128(199704)9:4%3C255::AID-CPE250%3E3.0.CO;2-2).

[4] D. Irony, S. Toledo, and A. Tiskin, “Communication Lower Bounds for Distributed-Memory Matrix Multiplication,” *Journal of Parallel and Distributed Computing*, vol. 64, no. 9, pp. 1017–1026, 2004, doi: [10.1016/j.jpdc.2004.03.021](https://doi.org/10.1016/j.jpdc.2004.03.021).

[5] MPI Forum, *MPI: A Message-Passing Interface Standard, Version 4.1*, Nov. 2023. [Official standard](https://www.mpi-forum.org/docs/mpi-4.1/).

[6] T. Hoefler, A. Lumsdaine, and W. Rehm, “Implementation and Performance Analysis of Non-Blocking Collective Operations for MPI,” in *Proc. ACM/IEEE Conference on Supercomputing (SC ’07)*, 2007, pp. 1–10, doi: [10.1145/1362622.1362692](https://doi.org/10.1145/1362622.1362692).

[7] A. Denis, J. Jaeger, E. Jeannot, and F. Reynier, “A Methodology for Assessing Computation/Communication Overlap of MPI Nonblocking Collectives,” *Concurrency and Computation: Practice and Experience*, vol. 34, no. 22, e7168, 2022, doi: [10.1002/cpe.7168](https://doi.org/10.1002/cpe.7168).

[8] L. S. Blackford et al., *ScaLAPACK Users’ Guide*. Philadelphia, PA: SIAM, 1997. [Netlib edition](https://www.netlib.org/scalapack/slug/).

# Appendix A – Literature Survey Proofs

Attach the title page or publisher/repository record for each reference: [1] Montana State thesis repository; [2] ACM SPAA ’94 proceedings/DOI record; [3] Wiley article record; [4] Elsevier journal record; [5] MPI Forum standard cover; [6] ACM/IEEE SC ’07 proceedings record; [7] Wiley article record; [8] SIAM/Netlib guide title page. Confirm college rules for attaching full papers; bibliographic proof pages may be sufficient.

# Appendix B – Source Code Link, README, Raw CSV and Demo Details

- Repository URL: [TO BE COMPLETED].
- Current inspected commit: `ca858a1687f8126e95b3ec75ebcc3f922c0b23ab` (update to final submission commit if changed).
- README: `README.md`.
- Sources: `src/oned_gemm.c`, `src/oned_gemm_double.c`, `src/summa.c`, `src/summa_double.c`, `src/cannon.c`, `src/cannon_double.c`, `src/summa_overlap.c`, `src/summa_overlap_double.c`, `src/cannon_overlap.c`, `src/cannon_overlap_double.c`.
- Dependencies: MSYS2 UCRT64 `mpicc`, Microsoft MPI runtime, OpenBLAS CBLAS headers/library, C math library. Python/pandas/matplotlib are used for graphs.
- Example compile:

```powershell
mpicc -O2 src\summa_double.c `
  -IC:/msys64/ucrt64/include/openblas `
  -LC:/msys64/ucrt64/lib -lopenblas -lm `
  -o summa_double.exe
```

- Example run and thread settings:

```powershell
$env:Path = "C:\msys64\ucrt64\bin;C:\Program Files\Microsoft MPI\Bin;" + $env:Path
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:GOTO_NUM_THREADS = '1'
$env:OMP_DYNAMIC = 'FALSE'
mpiexec -n 4 .\summa_double.exe 1024
```

- Original raw CSVs: `results/oned_gemm_float.csv`, `results/oned_gemm_double.csv`, `results/summa_float.csv`, `results/summa_double.csv`, `results/cannon_float.csv`, `results/cannon_double.csv`, `results/summa_overlap_float.csv`, `results/summa_overlap_double.csv`, `results/cannon_overlap_float.csv`, `results/cannon_overlap_double.csv`.
- Per-program CSVs in `results/` each contain the original 16 Windows records plus four Mac 8192 records; `results/benchmark_summary.csv` has 200 rows.
- Summary/logs: `results/benchmark_summary.csv`, `results/documentation/benchmark_summary.md`, `results/benchmark_logs/`.
- PNG graphs: `results/graphs/`; discussion: `results/documentation/graph_analysis.md`.
- Campaign automation: `scripts/run_benchmarks.ps1`; graph generation: `scripts/generate_graphs.py`.
- Separate 8192 Mac run: `results/8192_experiment/` contains outcomes, run metadata, README, and logs.

**Suggested 10-minute demonstration:** 1 min: define C=AB and the comparison; 2 min: explain row versus 2D blocks and SUMMA/Cannon communication; 2 min: run `mpiexec -n 4 .\summa_double.exe 512` and show timing/verification; 2 min: show nonblocking calls, computation window, wait and buffer swap; 2 min: compare one matched size/precision across P from CSV/graph and explain P=1 MPI speedup; 1 min: explain that local Windows 8192 was skipped and a separate Mac completed 40 runs. Label any prerecorded output; do not present it as a new run.

**Reproduce the Windows graphs:** from repository root, use Python with pandas and matplotlib, then run `python scripts\generate_graphs.py`. It validates all 200 rows in the ten CSVs and `benchmark_summary.csv`, but plots only the 160 Windows rows for sizes 512–4096; the 8192 Mac points are excluded to avoid mixing hardware. PNG files are written under `results/graphs/`, and updated analysis goes to `results/documentation/graph_analysis.md`.

# Appendix C – Weekly Progress Log and Individual Contribution

No dated weekly history was found in the inspected repository. Complete from actual records.

| Week/date | Work completed | Evidence (commit/file/run) | Blockers | Status | Contributor |
|---|---|---|---|---|---|
| [WEEK 1] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [NAME/USN] |
| [WEEK 2] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [NAME/USN] |
| [WEEK 3] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [NAME/USN] |
| [Add rows] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [NAME/USN] |
