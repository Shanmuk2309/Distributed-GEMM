
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include <math.h>

int main(int argc, char *argv[]) {

    int rank, size;
    int n = 512;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: %s <matrix_size> (512, 1024, 2048, 4096, or 8192)\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *end = NULL;
    long requested_n = strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || requested_n > 8192 || requested_n < 512 ||
        (requested_n != 512 && requested_n != 1024 && requested_n != 2048 &&
         requested_n != 4096 && requested_n != 8192)) {
        if (rank == 0)
            printf("Invalid matrix size. Use 512, 1024, 2048, 4096, or 8192.\n");

        MPI_Finalize();
        return 1;
    }
    n = (int)requested_n;

    size_t total_elements = (size_t)n * n;

    // Distribute rows as evenly as possible
    int local_rows = n / size;

    if (rank < n % size)
        local_rows++;

    int *sendcounts = NULL;
    int *displs = NULL;

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    double *C_ref = NULL;

    // Root process allocates complete matrices
    if (rank == 0) {

        A = malloc(total_elements * sizeof(double));
        B = malloc(total_elements * sizeof(double));
        C = malloc(total_elements * sizeof(double));
        C_ref = malloc(total_elements * sizeof(double));

        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        if (!A || !B || !C || !C_ref ||
            !sendcounts || !displs) {

            printf("Root memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Fixed random seed
        srand(42);

        for (size_t i = 0; i < total_elements; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }

        // Calculate row distribution
        int offset = 0;

        for (int i = 0; i < size; i++) {

            int rows = n / size + (i < n % size);

            sendcounts[i] = rows * n;
            displs[i] = offset;

            offset += sendcounts[i];
        }
    }

    // Local memory allocation
    double *local_A = malloc(
        (size_t)local_rows * n * sizeof(double));

    double *local_C = calloc(
        (size_t)local_rows * n, sizeof(double));

    if (!local_A || !local_C) {
        printf("Process %d: Local memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Non-root processes allocate B
    if (rank != 0) {

        B = malloc(total_elements * sizeof(double));

        if (!B) {
            printf("Process %d: B allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double total_start = MPI_Wtime();

    double comm_time = 0.0;
    double comp_time = 0.0;

    // 1. Scatter rows of A
    double t = MPI_Wtime();

    MPI_Scatterv(
        A, sendcounts, displs, MPI_DOUBLE,
        local_A, local_rows * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // 2. Broadcast complete matrix B
    MPI_Bcast(
        B, (int)total_elements, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    comm_time += MPI_Wtime() - t;

    // 3. Local matrix multiplication (i-k-j)
    t = MPI_Wtime();

    for (int i = 0; i < local_rows; i++) {

        for (int k = 0; k < n; k++) {

            double a = local_A[i * n + k];

            for (int j = 0; j < n; j++) {

                local_C[i * n + j] +=
                    a * B[k * n + j];
            }
        }
    }

    comp_time += MPI_Wtime() - t;

    // 4. Gather local C rows
    t = MPI_Wtime();

    MPI_Gatherv(
        local_C, local_rows * n, MPI_DOUBLE,
        C, sendcounts, displs, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    comm_time += MPI_Wtime() - t;

    double total_time = MPI_Wtime() - total_start;

    // 5. Find maximum time across all processes
    double max_total, max_comm, max_comp;

    MPI_Reduce(
        &total_time, &max_total, 1,
        MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
    );

    MPI_Reduce(
        &comm_time, &max_comm, 1,
        MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
    );

    MPI_Reduce(
        &comp_time, &max_comp, 1,
        MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
    );

    // 6. OpenBLAS reference calculation and verification
    if (rank == 0) {

        double operations = 2.0 * n * n * n;
        double gflops = operations / (max_total * 1e9);

        // Calculate reference matrix using OpenBLAS
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            n, n, n,
            1.0,
            A, n,
            B, n,
            0.0,
            C_ref, n
        );

        // Compare complete matrices
        double max_abs_error = 0.0;
        double max_rel_error = 0.0;

        for (size_t i = 0; i < total_elements; i++) {

            double abs_error = fabs(C[i] - C_ref[i]);

            double denominator =
                fmax(fabs(C_ref[i]), 1e-12);

            double rel_error = abs_error / denominator;

            if (abs_error > max_abs_error)
                max_abs_error = abs_error;

            if (rel_error > max_rel_error)
                max_rel_error = rel_error;
        }

        // Display benchmark results
        printf("\n===== 1D DISTRIBUTED GEMM (DOUBLE) =====\n");

        printf("Matrix size        : %d x %d\n", n, n);
        printf("Number of processes: %d\n", size);
        printf("Precision          : Double\n");

        printf("Execution time     : %.6f seconds\n", max_total);
        printf("Communication time : %.6f seconds\n", max_comm);
        printf("Computation time   : %.6f seconds\n", max_comp);

        printf("Performance        : %.4f GFLOP/s\n", gflops);

        printf("C[0][0]            : %.10f\n", C[0]);

        printf("\n----- OpenBLAS Verification -----\n");

        printf("Maximum absolute error : %.10e\n",
               max_abs_error);

        printf("Maximum relative error : %.10e\n",
               max_rel_error);

        if (max_rel_error < 1e-4)
            printf("Verification           : PASSED\n");
        else
            printf("Verification           : FAILED\n");

        printf("========================================\n");
    }

    // 7. Free memory
    free(local_A);
    free(local_C);
    free(B);

    if (rank == 0) {

        free(A);
        free(C);
        free(C_ref);

        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();

    return 0;
}
