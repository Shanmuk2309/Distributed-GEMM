
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int rank, size, n = 512;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc > 1) n = atoi(argv[1]);

    int q = (int)sqrt((double)size);

    if (n <= 0 || q * q != size) {
        if (rank == 0)
            printf("Use 1, 4, 9 or 16 processes and positive n.\n");
        MPI_Finalize();
        return 1;
    }

    int b = (n + q - 1) / q;
    int N = b * q;

    size_t bs = (size_t)b * b;
    size_t ms = (size_t)N * N;

    int dims[2] = {q, q};
    int periods[2] = {0, 0};

    MPI_Comm grid, row_comm, col_comm;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims,
                    periods, 0, &grid);

    int coords[2];
    MPI_Cart_coords(grid, rank, 2, coords);

    int remain_row[2] = {0, 1};
    int remain_col[2] = {1, 0};

    MPI_Cart_sub(grid, remain_row, &row_comm);
    MPI_Cart_sub(grid, remain_col, &col_comm);

    float *A = NULL, *B = NULL;
    float *packedA = NULL, *packedB = NULL;
    float *gatherC = NULL;

    if (rank == 0) {

        A = calloc(ms, sizeof(float));
        B = calloc(ms, sizeof(float));

        packedA = malloc((size_t)size * bs * sizeof(float));
        packedB = malloc((size_t)size * bs * sizeof(float));
        gatherC = malloc((size_t)size * bs * sizeof(float));

        if (!A || !B || !packedA || !packedB || !gatherC)
            MPI_Abort(grid, 1);

        srand(42);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[(size_t)i * N + j] =
                    (float)rand() / RAND_MAX;

                B[(size_t)i * N + j] =
                    (float)rand() / RAND_MAX;
            }
        }

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

    float *localA = malloc(bs * sizeof(float));
    float *localB = malloc(bs * sizeof(float));
    float *localC = calloc(bs, sizeof(float));

    // Two buffers for double buffering
    float *panelA[2];
    float *panelB[2];

    for (int i = 0; i < 2; i++) {
        panelA[i] = malloc(bs * sizeof(float));
        panelB[i] = malloc(bs * sizeof(float));
    }

    if (!localA || !localB || !localC ||
        !panelA[0] || !panelA[1] ||
        !panelB[0] || !panelB[1])
        MPI_Abort(grid, 1);

    MPI_Barrier(grid);

    double total_start = MPI_Wtime();
    double comm_time = 0.0;
    double comp_time = 0.0;

    double t = MPI_Wtime();

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

    comm_time += MPI_Wtime() - t;

    // Start the first panel broadcast
    int current = 0;

    if (coords[1] == 0)
        memcpy(panelA[current], localA, bs * sizeof(float));

    if (coords[0] == 0)
        memcpy(panelB[current], localB, bs * sizeof(float));

    MPI_Request req[2];

    t = MPI_Wtime();

    MPI_Ibcast(
        panelA[current], (int)bs, MPI_FLOAT,
        0, row_comm, &req[0]
    );

    MPI_Ibcast(
        panelB[current], (int)bs, MPI_FLOAT,
        0, col_comm, &req[1]
    );

    double init_time = MPI_Wtime() - t;

    t = MPI_Wtime();
    MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    double wait_time = MPI_Wtime() - t;

    comm_time += init_time + wait_time;

    // SUMMA with communication-computation overlap
    for (int k = 0; k < q; k++) {

        int next = 1 - current;

        double init_next = 0.0;
        double wait_next = 0.0;

        if (k + 1 < q) {

            // Prepare the next panel in the alternate buffers
            if (coords[1] == k + 1)
                memcpy(panelA[next], localA, bs * sizeof(float));

            if (coords[0] == k + 1)
                memcpy(panelB[next], localB, bs * sizeof(float));

            // Begin communication of next panels
            t = MPI_Wtime();

            MPI_Ibcast(
                panelA[next], (int)bs, MPI_FLOAT,
                k + 1, row_comm, &req[0]
            );

            MPI_Ibcast(
                panelB[next], (int)bs, MPI_FLOAT,
                k + 1, col_comm, &req[1]
            );

            init_next = MPI_Wtime() - t;
        }

        // Compute using current panels while next panels transfer
        t = MPI_Wtime();

        for (int i = 0; i < b; i++) {
            for (int x = 0; x < b; x++) {

                float a = panelA[current][i * b + x];

                for (int j = 0; j < b; j++) {
                    localC[i * b + j] +=
                        a * panelB[current][x * b + j];
                }
            }
        }

        comp_time += MPI_Wtime() - t;

        if (k + 1 < q) {

            // Ensure next panels have arrived
            t = MPI_Wtime();
            MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
            wait_next = MPI_Wtime() - t;

            comm_time += init_next + wait_next;

            current = next;
        }
    }

    t = MPI_Wtime();

    MPI_Gather(
        localC, (int)bs, MPI_FLOAT,
        gatherC, (int)bs, MPI_FLOAT,
        0, grid
    );

    comm_time += MPI_Wtime() - t;

    double total_time = MPI_Wtime() - total_start;

    double max_total, max_comm, max_comp;

    MPI_Reduce(&total_time, &max_total, 1,
               MPI_DOUBLE, MPI_MAX, 0, grid);

    MPI_Reduce(&comm_time, &max_comm, 1,
               MPI_DOUBLE, MPI_MAX, 0, grid);

    MPI_Reduce(&comp_time, &max_comp, 1,
               MPI_DOUBLE, MPI_MAX, 0, grid);

    if (rank == 0) {

        double gflops =
            (2.0 * n * n * n) / (max_total * 1e9);

        printf("\n======= SUMMA WITH OVERLAP =======\n");
        printf("Matrix size        : %d x %d\n", n, n);
        printf("Process grid       : %d x %d\n", q, q);
        printf("Processes          : %d\n", size);
        printf("Execution time     : %.6f seconds\n", max_total);
        printf("Communication time : %.6f seconds\n", max_comm);
        printf("Computation time   : %.6f seconds\n", max_comp);
        printf("Performance        : %.4f GFLOP/s\n", gflops);
        printf("C[0][0]            : %.6f\n", gatherC[0]);
        printf("==================================\n");
    }

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
