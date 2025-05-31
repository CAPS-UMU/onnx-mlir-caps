//
// Created by shreya on 30/05/25.
//
// main.c
#include <stdio.h>

// Declare the external matmul function
void matmul(long M, long N, long K, double *C, double *A, double *B);

int main() {
    long M = 2, K = 3, N = 2;
    double A[6] = { 1, 2, 3, 4, 5, 6 };
    double B[6] = { 7, 8, 9, 10, 11, 12 };
    double C[4];

    matmul(M, N, K, C, A, B);

    printf("Result matrix C:\n");
    for (long i = 0; i < M; i++) {
        for (long j = 0; j < N; j++) {
            printf("%f ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}
