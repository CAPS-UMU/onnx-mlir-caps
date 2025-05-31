// matmul.c
#include <stdio.h>

// Declare the matmul function (definition below)
void matmul(long M, long N, long K, double *C, double *A, double *B);

// Matrix multiplication using EPI intrinsics
void matmul(long M, long N, long K, double *C, double *A, double *B) {
    for (long i = 0; i < M; i++) {
        for (long j = 0; j < N; j++) {
            double sum = 0.0;
            for (long k = 0; k < K;) {
                long gvl = __builtin_epi_vsetvl(K - k, __epi_e64, __epi_m1);
                __epi_1xf64 vA = __builtin_epi_vload_1xf64(&A[i * K + k], gvl);

                double temp[gvl];
                for (long kk = 0; kk < gvl; kk++) {
                    temp[kk] = B[(k + kk) * N + j];
                }

                __epi_1xf64 vB = __builtin_epi_vload_1xf64(temp, gvl);
                __epi_1xf64 vProd = __builtin_epi_vfmul_1xf64(vA, vB, gvl);

                double partial[gvl];
                __builtin_epi_vstore_1xf64(partial, vProd, gvl);

                for (long kk = 0; kk < gvl; kk++) {
                    sum += partial[kk];
                }
                k += gvl;
            }
            C[i * N + j] = sum;
        }
    }
}
//
// Created by shreya on 30/05/25.
//
