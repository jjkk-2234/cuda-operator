import torch
import cuda_add

a = torch.randn(10_000_000, device='cuda')
b = torch.randn(10_000_000, device='cuda')
c = cuda_add.add(a, b)          # 带 autograd
c_no_grad = cuda_add.add_forward(a, b)  # 纯前向，稍快