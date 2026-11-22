
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <cblas.h>

int main(int argc, char *argv[]) {

    int rank, size, n = 512;

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
            printf("Use 1, 4, 9 or 16 processes and positive n.\n");

        MPI_Finalize();
        return 1;
    }

    // Pad matrix for equal-sized blocks
    int b = (n + q - 1) / q;
    int N = b * q;

    size_t bs = (size_t)b * b;
    size_t ms = (size_t)N * N;

    // Create 2D Cartesian grid
    int dims[2] = {q, q};
    int periods[2] = {0, 0};

    MPI_Comm grid, row_comm, col_comm;

    MPI_Cart_create(
        MPI_COMM_WORLD,
        2,
        dims,
        periods,
        0,
        &grid
    );

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);

    // Create row and column communicators
    int remain_row[2] = {0, 1};
    int remain_col[2] = {1, 0};

    MPI_Cart_sub(grid, remain_row, &row_comm);
    MPI_Cart_sub(grid, remain_col, &col_comm);

    double *A = NULL;
    double *B = NULL;

    double *packedA = NULL;
    double *packedB = NULL;

    double *gatherC = NULL;

    // OpenBLAS reference matrices
    double *A_ref = NULL;
    double *B_ref = NULL;
    double *C_ref = NULL;

    if (rank == 0) {

        A = calloc(ms, sizeof(double));
        B = calloc(ms, sizeof(double));

        packedA = malloc((size_t)size * bs * sizeof(double));
        packedB = malloc((size_t)size * bs * sizeof(double));

        gatherC = malloc((size_t)size * bs * sizeof(double));

        A_ref = malloc((size_t)n * n * sizeof(double));
        B_ref = malloc((size_t)n * n * sizeof(double));
        C_ref = malloc((size_t)n * n * sizeof(double));

        if (!A || !B || !packedA || !packedB || !gatherC ||
            !A_ref || !B_ref || !C_ref) {

            printf("Root allocation failed\n");
            MPI_Abort(grid, 1);
        }

        srand(42);

        // Generate matrices using fixed seed
        for (int i = 0; i < n; i++) {

            for (int j = 0; j < n; j++) {

                A[(size_t)i * N + j] =
                    (double)rand() / RAND_MAX;

                B[(size_t)i * N + j] =
                    (double)rand() / RAND_MAX;

                // Save original matrices for OpenBLAS
                A_ref[(size_t)i * n + j] =
                    A[(size_t)i * N + j];

                B_ref[(size_t)i * n + j] =
                    B[(size_t)i * N + j];
            }
        }

        // Pack matrix blocks
        for (int r = 0; r < q; r++) {

            for (int c = 0; c < q; c++) {

                int dest = r * q + c;

                for (int i = 0; i < b; i++) {

                    memcpy(
                        packedA + (size_t)dest * bs + (size_t)i * b,
                        A + (size_t)(r * b + i) * N + c * b,
                        b * sizeof(double)
                    );

                    memcpy(
                        packedB + (size_t)dest * bs + (size_t)i * b,
                        B + (size_t)(r * b + i) * N + c * b,
                        b * sizeof(double)
                    );
                }
            }
        }
    }

    // Allocate local matrix blocks
    double *localA = malloc(bs * sizeof(double));
    double *localB = malloc(bs * sizeof(double));
    double *localC = calloc(bs, sizeof(double));

    // Double buffers
    double *panelA[2];
    double *panelB[2];

    for (int i = 0; i < 2; i++) {

        panelA[i] = malloc(bs * sizeof(double));
        panelB[i] = malloc(bs * sizeof(double));
    }

    if (!localA || !localB || !localC ||
        !panelA[0] || !panelA[1] ||
        !panelB[0] || !panelB[1]) {

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
        packedA,
        (int)bs,
        MPI_DOUBLE,
        localA,
        (int)bs,
        MPI_DOUBLE,
        0,
        grid
    );

    MPI_Scatter(
        packedB,
        (int)bs,
        MPI_DOUBLE,
        localB,
        (int)bs,
        MPI_DOUBLE,
        0,
        grid
    );

    comm_time += MPI_Wtime() - t;

    // Start first panel broadcast
    int current = 0;

    if (coords[1] == 0)
        memcpy(panelA[current], localA, bs * sizeof(double));

    if (coords[0] == 0)
        memcpy(panelB[current], localB, bs * sizeof(double));

    MPI_Request req[2];
    MPI_Status statuses[2];

    t = MPI_Wtime();

    MPI_Ibcast(
        panelA[current],
        (int)bs,
        MPI_DOUBLE,
        0,
        row_comm,
        &req[0]
    );

    MPI_Ibcast(
        panelB[current],
        (int)bs,
        MPI_DOUBLE,
        0,
        col_comm,
        &req[1]
    );

    double init_time = MPI_Wtime() - t;

    t = MPI_Wtime();

    MPI_Waitall(2, req, statuses);

    double wait_time = MPI_Wtime() - t;

    comm_time += init_time + wait_time;

    // SUMMA with communication-computation overlap
    for (int k = 0; k < q; k++) {

        int next = 1 - current;

        double init_next = 0.0;
        double wait_next = 0.0;

        if (k + 1 < q) {

            // Prepare next panels
            if (coords[1] == k + 1) {

                memcpy(
                    panelA[next],
                    localA,
                    bs * sizeof(double)
                );
            }

            if (coords[0] == k + 1) {

                memcpy(
                    panelB[next],
                    localB,
                    bs * sizeof(double)
                );
            }

            // Start nonblocking broadcasts
            t = MPI_Wtime();

            MPI_Ibcast(
                panelA[next],
                (int)bs,
                MPI_DOUBLE,
                k + 1,
                row_comm,
                &req[0]
            );

            MPI_Ibcast(
                panelB[next],
                (int)bs,
                MPI_DOUBLE,
                k + 1,
                col_comm,
                &req[1]
            );

            init_next = MPI_Wtime() - t;
        }

        // Compute current panels while next panels are transferring
        t = MPI_Wtime();

        for (int i = 0; i < b; i++) {

            for (int x = 0; x < b; x++) {

                double a = panelA[current][i * b + x];

                for (int j = 0; j < b; j++) {

                    localC[i * b + j] +=
                        a * panelB[current][x * b + j];
                }
            }
        }

        comp_time += MPI_Wtime() - t;

        if (k + 1 < q) {

            // Wait for next panels
            t = MPI_Wtime();

            MPI_Waitall(2, req, statuses);

            wait_next = MPI_Wtime() - t;

            comm_time += init_next + wait_next;

            current = next;
        }
    }

    // Gather result blocks
    t = MPI_Wtime();

    MPI_Gather(
        localC,
        (int)bs,
        MPI_DOUBLE,
        gatherC,
        (int)bs,
        MPI_DOUBLE,
        0,
        grid
    );

    comm_time += MPI_Wtime() - t;

    double total_time = MPI_Wtime() - total_start;

    double max_total;
    double max_comm;
    double max_comp;

    MPI_Reduce(
        &total_time,
        &max_total,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        grid
    );

    MPI_Reduce(
        &comm_time,
        &max_comm,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        grid
    );

    MPI_Reduce(
        &comp_time,
        &max_comp,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        grid
    );

    if (rank == 0) {

        double gflops =
            (2.0 * n * n * n) / (max_total * 1e9);

        // OpenBLAS reference multiplication
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            n, n, n,
            1.0,
            A_ref, n,
            B_ref, n,
            0.0,
            C_ref, n
        );

        // Compare SUMMA overlap output with OpenBLAS
        double max_abs_error = 0.0;
        double max_rel_error = 0.0;

        for (int i = 0; i < n; i++) {

            for (int j = 0; j < n; j++) {

                int block_row = i / b;
                int block_col = j / b;

                int process_index =
                    block_row * q + block_col;

                int local_row = i % b;
                int local_col = j % b;

                size_t gather_index =
                    (size_t)process_index * bs +
                    (size_t)local_row * b +
                    local_col;

                double summa_value =
                    gatherC[gather_index];

                double reference_value =
                    C_ref[(size_t)i * n + j];

                double abs_error =
                    fabs((double)summa_value - reference_value);

                double denominator =
                    fmax(fabs((double)reference_value), 1e-7);

                double rel_error =
                    abs_error / denominator;

                if (abs_error > max_abs_error)
                    max_abs_error = abs_error;

                if (rel_error > max_rel_error)
                    max_rel_error = rel_error;
            }
        }

        printf("\n======= SUMMA WITH OVERLAP =======\n");

        printf("Matrix size        : %d x %d\n", n, n);
        printf("Padded size        : %d x %d\n", N, N);
        printf("Process grid       : %d x %d\n", q, q);
        printf("Processes          : %d\n", size);

        printf("Execution time     : %.6f seconds\n", max_total);
        printf("Communication time : %.6f seconds\n", max_comm);
        printf("Computation time   : %.6f seconds\n", max_comp);

        printf("Performance        : %.4f GFLOP/s\n", gflops);

        printf("C[0][0]            : %.6f\n", gatherC[0]);

        printf("\n----- OpenBLAS Verification -----\n");

        printf(
            "Maximum absolute error : %.10e\n",
            max_abs_error
        );

        printf(
            "Maximum relative error : %.10e\n",
            max_rel_error
        );

        if (max_rel_error < 1e-4)
            printf("Verification           : PASS\n");
        else
            printf("Verification           : FAIL\n");

        printf("==================================\n");

        free(A_ref);
        free(B_ref);
        free(C_ref);
    }

    // Free local memory
    free(localA);
    free(localB);
    free(localC);

    free(panelA[0]);
    free(panelA[1]);

    free(panelB[0]);
    free(panelB[1]);

    if (rank == 0) {

        free(A);
        free(B);

        free(packedA);
        free(packedB);

        free(gatherC);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid);

    MPI_Finalize();

    return 0;
}
