
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
            printf("Invalid size or process count. Use 1, 4, 9 or 16 processes.\n");

        MPI_Finalize();
        return 1;
    }

    // Pad matrix dimension for equal-sized blocks
    int b = (n + q - 1) / q;
    int N = b * q;

    size_t block_size = (size_t)b * b;
    size_t matrix_size = (size_t)N * N;
    size_t original_size = (size_t)n * n;

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

    int remain_row[2] = {0, 1};
    int remain_col[2] = {1, 0};

    MPI_Cart_sub(grid, remain_row, &row_comm);
    MPI_Cart_sub(grid, remain_col, &col_comm);

    double *A = NULL;
    double *B = NULL;
    double *packedA = NULL;
    double *packedB = NULL;
    double *gatherC = NULL;
    double *C_ref = NULL;

    // Root allocations and input generation
    if (rank == 0) {

        A = calloc(matrix_size, sizeof(double));
        B = calloc(matrix_size, sizeof(double));

        packedA = malloc(
            (size_t)size * block_size * sizeof(double)
        );

        packedB = malloc(
            (size_t)size * block_size * sizeof(double)
        );

        gatherC = malloc(
            (size_t)size * block_size * sizeof(double)
        );

        C_ref = malloc(original_size * sizeof(double));

        if (!A || !B || !packedA || !packedB ||
            !gatherC || !C_ref) {

            printf("Root allocation failed\n");
            MPI_Abort(grid, 1);
        }

        srand(42);

        // Generate same random values as serial/1D GEMM
        for (int i = 0; i < n * n; i++) {

            int r = i / n;
            int c = i % n;

            A[(size_t)r * N + c] =
                (double)rand() / RAND_MAX;

            B[(size_t)r * N + c] =
                (double)rand() / RAND_MAX;
        }

        // Pack matrix blocks for each process
        for (int r = 0; r < q; r++) {

            for (int c = 0; c < q; c++) {

                int dest = r * q + c;

                for (int i = 0; i < b; i++) {

                    memcpy(
                        packedA +
                            (size_t)dest * block_size +
                            (size_t)i * b,

                        A +
                            (size_t)(r * b + i) * N +
                            c * b,

                        b * sizeof(double)
                    );

                    memcpy(
                        packedB +
                            (size_t)dest * block_size +
                            (size_t)i * b,

                        B +
                            (size_t)(r * b + i) * N +
                            c * b,

                        b * sizeof(double)
                    );
                }
            }
        }
    }

    // Allocate local matrix blocks
    double *localA = malloc(block_size * sizeof(double));
    double *localB = malloc(block_size * sizeof(double));
    double *localC = calloc(block_size, sizeof(double));

    double *panelA = malloc(block_size * sizeof(double));
    double *panelB = malloc(block_size * sizeof(double));

    if (!localA || !localB || !localC ||
        !panelA || !panelB) {

        printf("Process %d: Local allocation failed\n", rank);
        MPI_Abort(grid, 1);
    }

    MPI_Barrier(grid);

    double total_start = MPI_Wtime();

    double comm_time = 0.0;
    double comp_time = 0.0;

    // Distribute A and B blocks
    double t = MPI_Wtime();

    MPI_Scatter(
        packedA,
        (int)block_size,
        MPI_DOUBLE,
        localA,
        (int)block_size,
        MPI_DOUBLE,
        0,
        grid
    );

    MPI_Scatter(
        packedB,
        (int)block_size,
        MPI_DOUBLE,
        localB,
        (int)block_size,
        MPI_DOUBLE,
        0,
        grid
    );

    comm_time += MPI_Wtime() - t;

    // SUMMA iterations
    for (int k = 0; k < q; k++) {

        // Select local panels
        if (coords[1] == k) {

            memcpy(
                panelA,
                localA,
                block_size * sizeof(double)
            );
        }

        if (coords[0] == k) {

            memcpy(
                panelB,
                localB,
                block_size * sizeof(double)
            );
        }

        // Broadcast A panel across process row
        t = MPI_Wtime();

        MPI_Bcast(
            panelA,
            (int)block_size,
            MPI_DOUBLE,
            k,
            row_comm
        );

        // Broadcast B panel down process column
        MPI_Bcast(
            panelB,
            (int)block_size,
            MPI_DOUBLE,
            k,
            col_comm
        );

        comm_time += MPI_Wtime() - t;

        // Multiply panels and accumulate local C
        t = MPI_Wtime();

        for (int i = 0; i < b; i++) {

            for (int x = 0; x < b; x++) {

                double a = panelA[i * b + x];

                for (int j = 0; j < b; j++) {

                    localC[i * b + j] +=
                        a * panelB[x * b + j];
                }
            }
        }

        comp_time += MPI_Wtime() - t;
    }

    // Gather final C blocks
    t = MPI_Wtime();

    MPI_Gather(
        localC,
        (int)block_size,
        MPI_DOUBLE,
        gatherC,
        (int)block_size,
        MPI_DOUBLE,
        0,
        grid
    );

    comm_time += MPI_Wtime() - t;

    double total_time = MPI_Wtime() - total_start;

    double max_total, max_comm, max_comp;

    MPI_Reduce(
        &total_time, &max_total,
        1, MPI_DOUBLE, MPI_MAX, 0, grid
    );

    MPI_Reduce(
        &comm_time, &max_comm,
        1, MPI_DOUBLE, MPI_MAX, 0, grid
    );

    MPI_Reduce(
        &comp_time, &max_comp,
        1, MPI_DOUBLE, MPI_MAX, 0, grid
    );

    // Root prints performance and verifies result
    if (rank == 0) {

        double c00 = gatherC[0];

        double operations = 2.0 * n * n * n;
        double gflops = operations / (max_total * 1e9);

        printf("\n========== SUMMA GEMM ==========\n");

        printf("Matrix size        : %d x %d\n", n, n);
        printf("Padded size        : %d x %d\n", N, N);
        printf("Process grid       : %d x %d\n", q, q);
        printf("Number of processes: %d\n", size);

        printf("Execution time     : %.6f seconds\n", max_total);
        printf("Communication time : %.6f seconds\n", max_comm);
        printf("Computation time   : %.6f seconds\n", max_comp);

        printf("Performance        : %.4f GFLOP/s\n", gflops);
        printf("C[0][0]            : %.6f\n", c00);

        // ====================================
        // OpenBLAS full matrix verification
        // ====================================

        double *A_original =
            malloc(original_size * sizeof(double));

        double *B_original =
            malloc(original_size * sizeof(double));

        if (!A_original || !B_original) {

            printf("Verification allocation failed\n");
            MPI_Abort(grid, 1);
        }

        // Extract original n x n matrices from padded A and B
        for (int i = 0; i < n; i++) {

            memcpy(
                A_original + (size_t)i * n,
                A + (size_t)i * N,
                n * sizeof(double)
            );

            memcpy(
                B_original + (size_t)i * n,
                B + (size_t)i * N,
                n * sizeof(double)
            );
        }

        // Calculate reference result using OpenBLAS
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            n, n, n,
            1.0,
            A_original, n,
            B_original, n,
            0.0,
            C_ref, n
        );

        double max_abs_error = 0.0;
        double max_rel_error = 0.0;

        // Compare SUMMA output against OpenBLAS
        for (int i = 0; i < n; i++) {

            for (int j = 0; j < n; j++) {

                // Determine the block containing C[i][j]
                int block_row = i / b;
                int block_col = j / b;

                // Process rank in the gathered buffer
                int process_index =
                    block_row * q + block_col;

                // Element position inside its local block
                int local_row = i % b;
                int local_col = j % b;

                // Correct index in gathered block storage
                size_t gather_index =
                    (size_t)process_index * block_size +
                    (size_t)local_row * b +
                    local_col;

                double summa_value = gatherC[gather_index];

                double ref_value =
                    C_ref[(size_t)i * n + j];

                double abs_error =
                    fabs((double)summa_value - ref_value);

                double denominator =
                    fmax(fabs((double)ref_value), 1e-7);

                double rel_error =
                    abs_error / denominator;

                if (abs_error > max_abs_error)
                    max_abs_error = abs_error;

                if (rel_error > max_rel_error)
                    max_rel_error = rel_error;
            }
        }

        printf("\n----- OpenBLAS Verification -----\n");

        printf(
            "Maximum absolute error : %.10e\n",
            max_abs_error
        );

        printf(
            "Maximum relative error : %.10e\n",
            max_rel_error
        );

        if (max_rel_error < 1e-4) {

            printf("Verification           : PASSED\n");

        } else {

            printf("Verification           : FAILED\n");
        }

        printf("================================\n");

        free(A_original);
        free(B_original);
    }

    // Free local memory
    free(localA);
    free(localB);
    free(localC);
    free(panelA);
    free(panelB);

    // Free root memory
    if (rank == 0) {

        free(A);
        free(B);
        free(packedA);
        free(packedB);
        free(gatherC);
        free(C_ref);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid);

    MPI_Finalize();

    return 0;
}
