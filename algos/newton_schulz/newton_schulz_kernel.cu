// newton_schulz_kernel.cu

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void newton_schulz_kernel(
    const scalar_t* __restrict__ G,
    scalar_t* __restrict__ X,
    scalar_t* __restrict__ buffer,
    const int N,
    const int M,
    const int steps,
    const scalar_t a,
    const scalar_t b,
    const scalar_t c) {
    // Shared memory for tiles (optional)
    // extern __shared__ scalar_t shared_data[];

    // Compute global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    // Guard against out-of-bounds threads
    if (row >= N || col >= M)
        return;

    // Load G into X as initial value
    X[row * M + col] = G[row * M + col];

    // Initialize buffer pointers
    scalar_t* A = buffer;            // Size N*N
    scalar_t* B = buffer + N * N;    // Size N*N

    // Synchronize to ensure all threads have loaded initial X
    __syncthreads();

    // Perform iterations
    for (int step = 0; step < steps; ++step) {
        // Compute A = X @ X^T
        scalar_t sum_A = 0;
        for (int k = 0; k < M; ++k) {
            scalar_t x_ik = X[row * M + k];
            scalar_t x_jk = X[col * M + k]; // Note: Transposed
            sum_A += x_ik * x_jk;
        }
        // Store A[row, col]
        if (row < N && col < N) {
            A[row * N + col] = sum_A;
        }

        __syncthreads();

        // Compute B = b * A + c * A @ A
        scalar_t sum_B = 0;
        if (row < N && col < N) {
            // Compute (A @ A)[row, col]
            scalar_t sum_AA = 0;
            for (int k = 0; k < N; ++k) {
                sum_AA += A[row * N + k] * A[k * N + col];
            }
            // Compute B[row, col]
            B[row * N + col] = b * A[row * N + col] + c * sum_AA;
        }

        __syncthreads();

        // Update X = a * X + B @ X
        scalar_t sum_X = 0;
        for (int k = 0; k < N; ++k) {
            scalar_t b_ik = B[row * N + k];
            scalar_t x_kj = X[k * M + col];
            sum_X += b_ik * x_kj;
        }
        // Update X[row, col]
        X[row * M + col] = a * X[row * M + col] + sum_X;

        __syncthreads();
    }
}
