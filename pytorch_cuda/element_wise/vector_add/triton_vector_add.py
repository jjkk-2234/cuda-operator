import torch
import triton
import triton.language as tl
import time
# 用于自动调优BLOCK_SIZE
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N']
)

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 类似cuda中的blockIdx.x
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    # tl.arange部分类似cuda中的threadIdx.x
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.is_contiguous() and y.is_contiguous()
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, N)
    return out

def verify():
    print("========== 正确性校验（多形状，含非整除 N） ==========")
    for n in [1, 2, 3, 100, 1023, 4096, 100000]:
        a = torch.randn(n, device='cuda', dtype=torch.float32)
        b = torch.randn(n, device='cuda', dtype=torch.float32)
        c = add(a, b)
        ref = a + b
        ok = torch.allclose(c, ref)
        print(f"N={n}: allclose={ok}")
        assert ok, f"N={n} 校验失败"
    print("正确性校验：ALL PASS\n")

def main():
    verify()
    N = 1 << 26
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)

    c = add(a, b)
    torch.cuda.synchronize()
    start = time.time()
    c = add(a, b)
    torch.cuda.synchronize()
    end = time.time()

    elapsed_time_ms = (end - start) * 1000

    ms = elapsed_time_ms
    bytes_total = 3 * N * 4
    bw = (bytes_total / 1e9) / (ms / 1000.0)
    print(f"Triton Add: {ms:.3f} ms, Bandwidth: {bw:.3f} GB/s")

main()