#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>

torch::Tensor matmul(torch::Tensor a, torch::Tensor b) {
    std::cout << "Tensor a is is on: " << a.device() 
    assert(a.device().type() == torch::kCUDA);
    assert(b.device().type() == torch::kCUDA);
    assert(a.dtype() == torch::kByte);
    assert(b.dtype() == torch::kByte);

    // ensure that matmul sizes are appropriate
    const auto n_rows_A = a.size(0);
    const auto n_rows_B = b.size(0);

    const auto n_cols_A = a.size(1);
    const auto n_cols_B = b.size(1);

    assert(n_cols_A == n_cols_B);
    
}
