
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int rank, size;
    int n = 512;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    if (n <= 0) {
        if (rank == 0)
            printf("Invalid matrix size\n");
        MPI_Finalize();
        return 1;
    }

    // Distribute rows as evenly as possible
    int local_rows = n / size;
    if (rank < n % size)
        local_rows++;

    int *sendcounts = NULL;
    int *displs = NULL;

    float *A = NULL;
    float *B = NULL;
    float *C = NULL;

    if (rank == 0) {
        A = malloc((size_t)n * n * sizeof(float));
        B = malloc((size_t)n * n * sizeof(float));
        C = malloc((size_t)n * n * sizeof(float));

        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        if (!A || !B || !C || !sendcounts || !displs) {
            printf("Root memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Fixed random seed
        srand(42);

        for (int i = 0; i < n * n; i++) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
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
    float *local_A = malloc(
        (size_t)local_rows * n * sizeof(float));

    float *local_C = calloc(
        (size_t)local_rows * n, sizeof(float));

    if (!local_A || !local_C) {
        printf("Process %d: Local memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank != 0) {
        B = malloc((size_t)n * n * sizeof(float));

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
        A, sendcounts, displs, MPI_FLOAT,
        local_A, local_rows * n, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    // 2. Broadcast complete matrix B
    MPI_Bcast(
        B, n * n, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    comm_time += MPI_Wtime() - t;

    // 3. Local matrix multiplication
    t = MPI_Wtime();

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;

            for (int k = 0; k < n; k++) {
                sum += local_A[i * n + k] * B[k * n + j];
            }

            local_C[i * n + j] = sum;
        }
    }

    comp_time += MPI_Wtime() - t;

    // 4. Gather local C rows
    t = MPI_Wtime();

    MPI_Gatherv(
        local_C, local_rows * n, MPI_FLOAT,
        C, sendcounts, displs, MPI_FLOAT,
        0, MPI_COMM_WORLD
    );

    comm_time += MPI_Wtime() - t;

    double total_time = MPI_Wtime() - total_start;

    // 5. Find maximum time across all processes
    double max_total, max_comm, max_comp;

    MPI_Reduce(&total_time, &max_total, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&comm_time, &max_comm, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&comp_time, &max_comp, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // 6. Display results
    if (rank == 0) {

        double operations = 2.0 * n * n * n;
        double gflops = operations / (max_total * 1e9);

        printf("\n===== 1D DISTRIBUTED GEMM =====\n");
        printf("Matrix size       : %d x %d\n", n, n);
        printf("Number of processes: %d\n", size);

        printf("Execution time    : %.6f seconds\n", max_total);
        printf("Communication time: %.6f seconds\n", max_comm);
        printf("Computation time  : %.6f seconds\n", max_comp);

        printf("Performance       : %.4f GFLOP/s\n", gflops);
        printf("C[0][0]           : %.6f\n", C[0]);

        printf("===============================\n");
    }

    free(local_A);
    free(local_C);
    free(B);

    if (rank == 0) {
        free(A);
        free(C);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();

    return 0;
}
