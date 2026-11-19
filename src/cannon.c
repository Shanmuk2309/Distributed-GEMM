
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int rank, size;
    int n = 512;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1)
        n = atoi(argv[1]);

    int q = (int)sqrt((double)size);

    if (n <= 0 || q * q != size) {
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

    if (rank == 0) {

        A = calloc(ms, sizeof(float));
        B = calloc(ms, sizeof(float));

        packedA = malloc((size_t)size * bs * sizeof(float));
        packedB = malloc((size_t)size * bs * sizeof(float));
        gatherC = malloc((size_t)size * bs * sizeof(float));

        if (!A || !B || !packedA || !packedB || !gatherC) {
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

    // Cannon multiplication and circular shifting
    for (int step = 0; step < q; step++) {

        t = MPI_Wtime();

        // Local block multiplication
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < b; k++) {

                float a = localA[i * b + k];

                for (int j = 0; j < b; j++) {

                    localC[i * b + j] +=
                        a * localB[k * b + j];
                }
            }
        }

        comp_time += MPI_Wtime() - t;

        // Circular shifts for the next iteration
        if (step < q - 1) {

            t = MPI_Wtime();

            // Shift A one position left
            MPI_Sendrecv_replace(
                localA, (int)bs, MPI_FLOAT,
                left, 2, right, 2,
                grid, MPI_STATUS_IGNORE
            );

            // Shift B one position upward
            MPI_Sendrecv_replace(
                localB, (int)bs, MPI_FLOAT,
                up, 3, down, 3,
                grid, MPI_STATUS_IGNORE
            );

            comm_time += MPI_Wtime() - t;
        }
    }

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

        printf("=================================\n");
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
