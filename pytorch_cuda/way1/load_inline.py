import torch
from torch.utils.cpp_extension import load_inline
cpp_src = """
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
"""

cuda_src = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int N = a.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
    return c;
}
"""

module = load_inline(
    name = "add_cuda", 
    cpp_sources = cpp_src, 
    cuda_sources = cuda_src,
    functions = ["add_cuda"],
    verbose = True
)

a = torch.randn(1000, device='cuda')
b = torch.randn(1000, device='cuda')
c = module.add_cuda(a, b)
print(torch.allclose(c, a + b))  # Should print True