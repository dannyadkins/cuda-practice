// newton_schulz_binding.cpp

#include <torch/extension.h>

void newton_schulz_cuda(
    torch::Tensor G,
    torch::Tensor X,
    int steps,
    float a,
    float b,
    float c) {
    const auto N = G.size(0);
    const auto M = G.size(1);

    // Allocate buffer for A and B
    auto buffer = torch::empty({2 * N * N}, G.options());

    // Determine block and grid sizes
    const int threads = 16;
    const dim3 block_dim(threads, threads);
    const dim3 grid_dim((M + threads - 1) / threads, (N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(G.scalar_type(), "newton_schulz_cuda", ([&] {
        newton_schulz_kernel<scalar_t><<<grid_dim, block_dim>>>(
            G.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            buffer.data_ptr<scalar_t>(),
            N,
            M,
            steps,
            static_cast<scalar_t>(a),
            static_cast<scalar_t>(b),
            static_cast<scalar_t>(c)
        );
    }));

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("newton_schulz_cuda", &newton_schulz_cuda, "Newton-Schulz CUDA kernel");
}
