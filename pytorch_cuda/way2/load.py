import torch
from torch.utils.cpp_extension import load

vector_add = load(
    name="vector_add",
    sources=["vector_add.cu"],
    verbose = True,
    extra_cuda_cflags=["-O3"]
)

x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')
print(vector_add.add(x, y))