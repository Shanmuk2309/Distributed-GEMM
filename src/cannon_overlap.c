
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cblas.h>

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
    if (end == argv[1] || *end != '\0' ||
        (requested_n != 512 && requested_n != 1024 && requested_n != 2048 &&
         requested_n != 4096 && requested_n != 8192)) {
        if (rank == 0)
            printf("Invalid matrix size. Use 512, 1024, 2048, 4096, or 8192.\n");
        MPI_Finalize();
        return 1;
    }
    n = (int)requested_n;

    int q = (int)sqrt((double)size);

    if (n <= 0 || q * q != size || q < 1 || q > 4) {
        if (rank == 0)
            printf("Use a positive matrix size and 1, 4, 9 or 16 processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Pad matrix for equal-sized blocks
    int b = (n + q - 1) / q;
    int N = b * q;

    size_t bs = (size_t)b * b;
    size_t ms = (size_t)N * N;

    // Create periodic 2D Cartesian grid
    int dims[2] = {q, q};
    int periods[2] = {1, 1};

    MPI_Comm grid;

    MPI_Cart_create(
        MPI_COMM_WORLD, 2, dims, periods, 0, &grid
    );

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);

    // Neighbours for circular shifts
    int left, right, up, down;

    MPI_Cart_shift(grid, 1, -1, &right, &left);
    MPI_Cart_shift(grid, 0, -1, &down, &up);

    float *A = NULL, *B = NULL;
    float *packedA = NULL, *packedB = NULL;
    float *gatherC = NULL;

    float *A_ref = NULL, *B_ref = NULL, *C_ref = NULL;

    if (rank == 0) {

        A = calloc(ms, sizeof(float));
        B = calloc(ms, sizeof(float));

        packedA = malloc((size_t)size * bs * sizeof(float));
        packedB = malloc((size_t)size * bs * sizeof(float));
        gatherC = malloc((size_t)size * bs * sizeof(float));

        A_ref = malloc((size_t)n * n * sizeof(float));
        B_ref = malloc((size_t)n * n * sizeof(float));
        C_ref = malloc((size_t)n * n * sizeof(float));

        if (!A || !B || !packedA || !packedB || !gatherC ||
            !A_ref || !B_ref || !C_ref) {
            printf("Root allocation failed\n");
            MPI_Abort(grid, 1);
        }

        srand(42);

        // Generate matrices with fixed seed
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {

                A[(size_t)i * N + j] =
                    (float)rand() / RAND_MAX;

                B[(size_t)i * N + j] =
                    (float)rand() / RAND_MAX;

                A_ref[(size_t)i * n + j] =
                    A[(size_t)i * N + j];

                B_ref[(size_t)i * n + j] =
                    B[(size_t)i * N + j];
            }
        }

        // Pack blocks for distribution
        for (int r = 0; r < q; r++) {
            for (int c = 0; c < q; c++) {

                int dest = r * q + c;

                for (int i = 0; i < b; i++) {

                    memcpy(
                        packedA + (size_t)dest * bs + (size_t)i * b,
                        A + (size_t)(r * b + i) * N + c * b,
                        b * sizeof(float)
                    );

                    memcpy(
                        packedB + (size_t)dest * bs + (size_t)i * b,
                        B + (size_t)(r * b + i) * N + c * b,
                        b * sizeof(float)
                    );
                }
            }
        }
    }

    // Local blocks
    float *localA = malloc(bs * sizeof(float));
    float *localB = malloc(bs * sizeof(float));
    float *localC = calloc(bs, sizeof(float));

    if (!localA || !localB || !localC) {
        printf("Process %d: Allocation failed\n", rank);
        MPI_Abort(grid, 1);
    }

    MPI_Barrier(grid);

    double total_start = MPI_Wtime();
    double comm_time = 0.0;
    double comp_time = 0.0;

    double t = MPI_Wtime();

    // Distribute matrix blocks
    MPI_Scatter(
        packedA, (int)bs, MPI_FLOAT,
        localA, (int)bs, MPI_FLOAT,
        0, grid
    );

    MPI_Scatter(
        packedB, (int)bs, MPI_FLOAT,
        localB, (int)bs, MPI_FLOAT,
        0, grid
    );

    // Initial skewing:
    // Shift A left by process row coordinate
    for (int i = 0; i < coords[0]; i++) {

        MPI_Sendrecv_replace(
            localA, (int)bs, MPI_FLOAT,
            left, 0, right, 0,
            grid, MPI_STATUS_IGNORE
        );
    }

    // Shift B upward by process column coordinate
    for (int i = 0; i < coords[1]; i++) {

        MPI_Sendrecv_replace(
            localB, (int)bs, MPI_FLOAT,
            up, 1, down, 1,
            grid, MPI_STATUS_IGNORE
        );
    }

    comm_time += MPI_Wtime() - t;

    // Double-buffer blocks so the next shifts can progress during computation.
    float *nextA = malloc(bs * sizeof(float));
    float *nextB = malloc(bs * sizeof(float));
    if (!nextA || !nextB) {
        printf("Process %d: Double-buffer allocation failed\n", rank);
        MPI_Abort(grid, 1);
    }

    // Cannon multiplication with nonblocking next-block transfers.
    for (int step = 0; step < q; step++) {
        MPI_Request requests[4];
        MPI_Status statuses[4];
        int request_count = 0;
        double shift_start = 0.0;

        if (step < q - 1) {
            double post_start = MPI_Wtime();
            MPI_Irecv(nextA, (int)bs, MPI_FLOAT, right, 2, grid, &requests[request_count++]);
            MPI_Isend(localA, (int)bs, MPI_FLOAT, left, 2, grid, &requests[request_count++]);
            MPI_Irecv(nextB, (int)bs, MPI_FLOAT, down, 3, grid, &requests[request_count++]);
            MPI_Isend(localB, (int)bs, MPI_FLOAT, up, 3, grid, &requests[request_count++]);
            shift_start = MPI_Wtime() - post_start;
        }

        t = MPI_Wtime();
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < b; k++) {
                float a = localA[i * b + k];
                for (int j = 0; j < b; j++)
                    localC[i * b + j] += a * localB[k * b + j];
            }
        }
        comp_time += MPI_Wtime() - t;

        if (step < q - 1) {
            t = MPI_Wtime();
            MPI_Waitall(request_count, requests, statuses);
            comm_time += shift_start + MPI_Wtime() - t;
            float *swap = localA; localA = nextA; nextA = swap;
            swap = localB; localB = nextB; nextB = swap;
        }
    }

    free(nextA);
    free(nextB);

    // Collect output blocks
    t = MPI_Wtime();

    MPI_Gather(
        localC, (int)bs, MPI_FLOAT,
        gatherC, (int)bs, MPI_FLOAT,
        0, grid
    );

    comm_time += MPI_Wtime() - t;

    double total_time = MPI_Wtime() - total_start;

    double max_total, max_comm, max_comp;

    MPI_Reduce(
        &total_time, &max_total, 1,
        MPI_DOUBLE, MPI_MAX, 0, grid
    );

    MPI_Reduce(
        &comm_time, &max_comm, 1,
        MPI_DOUBLE, MPI_MAX, 0, grid
    );

    MPI_Reduce(
        &comp_time, &max_comp, 1,
        MPI_DOUBLE, MPI_MAX, 0, grid
    );

    if (rank == 0) {

        double operations = 2.0 * n * n * n;
        double gflops = operations / (max_total * 1e9);

        // OpenBLAS reference multiplication
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            n, n, n,
            1.0f,
            A_ref, n,
            B_ref, n,
            0.0f,
            C_ref, n
        );

        // Compare Cannon output with OpenBLAS
        double max_abs_error = 0.0;
        double max_rel_error = 0.0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {

                int block_row = i / b;
                int block_col = j / b;

                int process_index = block_row * q + block_col;

                int local_row = i % b;
                int local_col = j % b;

                size_t gather_index =
                    (size_t)process_index * bs +
                    (size_t)local_row * b +
                    local_col;

                float cannon_value = gatherC[gather_index];
                float reference_value = C_ref[(size_t)i * n + j];

                double abs_error =
                    fabsf(cannon_value - reference_value);

                double denominator =
                    fmaxf(fabsf(reference_value), 1e-7f);

                double rel_error = abs_error / denominator;

                if (abs_error > max_abs_error)
                    max_abs_error = abs_error;

                if (rel_error > max_rel_error)
                    max_rel_error = rel_error;
            }
        }

        printf("\n========== CANNON GEMM ==========\n");
        printf("Matrix size        : %d x %d\n", n, n);
        printf("Padded size        : %d x %d\n", N, N);
        printf("Process grid       : %d x %d\n", q, q);
        printf("Number of processes: %d\n", size);

        printf("Execution time     : %.6f seconds\n", max_total);
        printf("Communication time : %.6f seconds\n", max_comm);
        printf("Computation time   : %.6f seconds\n", max_comp);

        printf("Performance        : %.4f GFLOP/s\n", gflops);
        printf("C[0][0]            : %.6f\n", gatherC[0]);

        printf("\n----- OpenBLAS Verification -----\n");
        printf("Maximum absolute error : %.10e\n", max_abs_error);
        printf("Maximum relative error : %.10e\n", max_rel_error);

        if (max_rel_error < 1e-4) {
            printf("Verification           : PASS\n");
        } else {
            printf("Verification           : FAIL\n");
        }

        printf("=================================\n");

        free(A_ref);
        free(B_ref);
        free(C_ref);
    }

    free(localA);
    free(localB);
    free(localC);

    if (rank == 0) {
        free(A);
        free(B);
        free(packedA);
        free(packedB);
        free(gatherC);
    }

    MPI_Comm_free(&grid);

    MPI_Finalize();

    return 0;
}
