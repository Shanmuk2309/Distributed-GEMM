\
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 512

int main() {

    int n = N;

    float *A = malloc(n * n * sizeof(float));
    float *B = malloc(n * n * sizeof(float));
    float *C = calloc(n * n, sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed!\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    // Fixed seed for reproducible random values
    srand(42);

    // Initialize matrices A and B
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    // Start timing
    clock_t start = clock();

    // Matrix multiplication: C = A * B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] +=
                    A[i * n + k] * B[k * n + j];
            }
        }
    }

    // Stop timing
    clock_t end = clock();

    double time_taken =
        (double)(end - start) / CLOCKS_PER_SEC;

    double operations = 2.0 * n * n * n;

    double gflops =
        operations / (time_taken * 1e9);

    printf("Matrix size: %d x %d\n", n, n);
    printf("Execution time: %.6f seconds\n", time_taken);
    printf("Performance: %.4f GFLOP/s\n", gflops);

    printf("C[0][0] = %f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}

