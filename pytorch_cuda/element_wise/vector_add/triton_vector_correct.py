import torch
import triton
import triton.language as tl
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

# 用 CUDA event 计时：预热 + 多次重复取平均，避免 time.time() 包含的
# Python 启动/autotune 开销，得到真正的 kernel 执行时间
def benchmark_add(n_warmup=5, n_repeat=20):
    N = 1 << 26
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)

    # 预热（autotune 也在此期间完成并选好最优 config）
    for _ in range(n_warmup):
        add(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        add(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start.elapsed_time(end) / n_repeat
    return elapsed_time_ms, N

ms, N = benchmark_add()
# 计算有效带宽
bytes_total = 3 * N * 4
bw = (bytes_total / 1e9) / (ms / 1000.0)
print(f"Triton Add: {ms:.3f} ms, Bandwidth: {bw:.3f} GB/s")