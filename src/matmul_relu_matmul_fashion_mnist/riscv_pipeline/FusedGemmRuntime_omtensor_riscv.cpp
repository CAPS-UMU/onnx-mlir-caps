/**********************************************
 * IMPORT LIBRARIES ###########################
 **********************************************/
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

// ONNX-MLIR Runtime API
#include "OnnxMlirRuntime.h"
// Helper for element‐count
#include <numeric> // For std::accumulate

// RISCV Library for Matrix Multiplication (assumed compiled with this file)
// Version: matmu_rvvlib v0.1 (example)
#ifndef RISC_MATMUL_SRC
#define RISC_MATMUL_SRC "matmu_rvvlib.cpp"
#endif
#include RISC_MATMUL_SRC



/**********************************************
 * HELPER FUNCTION DEFINITIONS ################
 **********************************************/
/**
 * @brief Computes the product of the dimensions of a tensor.
 *
 * @param tensor Pointer to an OMTensor.
 * @return size_t The number of elements in the tensor.
 */
static size_t numElements(OMTensor* tensor) {
  size_t rank = omTensorGetRank(tensor);
  const int64_t* shape = omTensorGetShape(tensor);
  size_t prod = 1;
  for (size_t i = 0; i < rank; ++i)
    prod *= shape[i];
  return prod;
}

/**********************************************
 * MAIN RUNTIME FUNCTION DEFINITION ###########
 **********************************************/
/**
 * @brief Executes the fused Gemm+ReLU operation on RISC-V via matmu_rvvlib.
 *
 * @param Y_omTensor    Output OMTensor (shape [M,N]).
 * @param A_omTensor    Input OMTensor A (shape [M,K] if transA==0).
 * @param B_omTensor    Input OMTensor B (shape [K,N] if transB==0).
 * @param Bias_omTensor Bias OMTensor (shape [N], may be nullptr).
 * @param activation    Unused (ReLU is hardcoded).
 * @param alpha         Scaling factor for Gemm output.
 * @param beta          Scaling factor for bias.
 * @param domain_name   Unused.
 * @param onnx_node_name Unused.
 * @param transA        Transpose flag for A (0=no,1=yes).
 * @param transB        Transpose flag for B (0=no,1=yes).
 */
extern "C" void FusedGemm(
    OMTensor* Y_omTensor,
    OMTensor* A_omTensor,
    OMTensor* B_omTensor,
    OMTensor* Bias_omTensor,
    const char *activation,
    float alpha,
    float beta,
    const char *domain_name,
    const char *onnx_node_name,
    int64_t transA,
    int64_t transB
) {
  /**********************************************
   * VALIDATE INPUT TENSORS ####################
   **********************************************/
  if (!A_omTensor || !B_omTensor || !Y_omTensor) {
    fprintf(stdout, "Error: One or more input tensors are null.\n");
    return;
  }
  if (omTensorGetDataType(A_omTensor) != ONNX_TYPE_FLOAT ||
      omTensorGetDataType(B_omTensor) != ONNX_TYPE_FLOAT ||
      omTensorGetDataType(Y_omTensor) != ONNX_TYPE_FLOAT) {
    fprintf(stdout, "Error: Input tensors must be of type float.\n");
    return;
  }

  /**********************************************
   * RETRIEVE SHAPES & DATA PTRS ###############
   **********************************************/
  const int64_t* Y_shape = omTensorGetShape(Y_omTensor);
  int64_t M = Y_shape[0];
  int64_t N = Y_shape[1];
  float* A_ptr = static_cast<float*>(omTensorGetDataPtr(A_omTensor));
  float* B_ptr = static_cast<float*>(omTensorGetDataPtr(B_omTensor));
  float* Bias_ptr = Bias_omTensor
                      ? static_cast<float*>(omTensorGetDataPtr(Bias_omTensor))
                      : nullptr;
  float* Y_ptr = static_cast<float*>(omTensorGetDataPtr(Y_omTensor));

  /**********************************************
   * COMPUTE K and ALLOCATE TEMP DOUBLE BUFS ###
   **********************************************/
  size_t total_A = numElements(A_omTensor);      // M*K
  size_t K = total_A / static_cast<size_t>(M);
  size_t total_B = numElements(B_omTensor);      // K*N
  size_t total_Y = static_cast<size_t>(M) * N;   // M*N

  // Convert A and B to double for matmul
  std::vector<double> A_d(total_A), B_d(total_B), Y_d(total_Y);
  for (size_t i = 0; i < total_A; ++i)
    A_d[i] = static_cast<double>(A_ptr[i]);
  for (size_t i = 0; i < total_B; ++i)
    B_d[i] = static_cast<double>(B_ptr[i]);

  /**********************************************
   * INVOKE RISCV MATMUL #######################
   **********************************************/
  // matmul(M, N, K, C, A, B) -- C is output double*
  matmul(static_cast<long>(M),
         static_cast<long>(N),
         static_cast<long>(K),
         Y_d.data(),
         A_d.data(),
         B_d.data());

  /**********************************************
   * APPLY BIAS, SCALING, AND RELU (if not done in "matmul")
   **********************************************/
  for (size_t idx = 0; idx < total_Y; ++idx) {
    double acc = alpha * Y_d[idx];
    if (Bias_ptr) {
      // broadcast bias over batch: bias index = idx % N
      acc += beta * static_cast<double>( Bias_ptr[idx % N] );
    }
    // Write-back to Y_ptr, ReLU activation & cast back to float
    Y_ptr[idx] = static_cast<float>( std::max(acc, 0.0) );
  }
  

  fprintf(stdout, "FusedGemm: Completed RISC-V matmul + bias + ReLU.\\n");
}