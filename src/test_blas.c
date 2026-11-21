#include <stdio.h>
#include <cblas.h>

int main() {
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0};

    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        2, 2, 2,
        1.0f,
        A, 2,
        B, 2,
        0.0f,
        C, 2
    );

    printf("C =\n");
    printf("%.1f %.1f\n", C[0], C[1]);
    printf("%.1f %.1f\n", C[2], C[3]);

    return 0;
}
