import torch
import triton
import triton.language as tl
import time
import numpy as np

# 方案一：直接加法
@triton.jit
def matrix_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    ga = tl.load(a + offset, mask=mask)
    gb = tl.load(b + offset, mask=mask)
    gc = ga + gb
    tl.store(c + offset, gc, mask=mask)

def solve_triton_naive(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_elements = N * N
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    matrix_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)

# 方案二：Triton一维向量化 + 自动调优
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_WIDTH': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_WIDTH': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_WIDTH': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_WIDTH': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_WIDTH': 8}, num_warps=16),
    ],
    key=['n_elements'],
)

@triton.jit
def matrix_add_kernel_1d(a_ptr, b_ptr, c_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr, VEC_WIDTH: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    # None 就是 np.newaxis，用于添加新维度
    offsets = block_start + tl.arange(0, BLOCK_SIZE)[:, None] * VEC_WIDTH + tl.arange(0, VEC_WIDTH)[None, :]
    offsets = tl.reshape(offsets, (BLOCK_SIZE * VEC_WIDTH,))

    mask = offsets < n_elements

    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = a_vals + b_vals

    tl.store(c_ptr + offsets, c_vals, mask=mask)

def solve_triton_1d(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    n_elements = N * N
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * meta['VEC_WIDTH']),)
    matrix_add_kernel_1d[grid](a, b, c, n_elements)

# 方案三：Triton二维向量化 + 自动调优
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N'],
)

@triton.jit
def matrix_add_kernel_2d(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # base 是矩阵 a 的首地址，shape是完整的矩阵形状
    # stride 是各维度的内存跨步，这是连接逻辑坐标和物理地址的关键（ND格式）
    # offfsets 是子块左上角的起始坐标
    # block_shape 是子块的形状
    # order 是内存布局顺序，(1, 0) 表示按行优先存储
    a_block_ptr = tl.make_block_ptr(
       base = a_ptr,
       shape = (n_elements, n_elements),
       strides = (n_elements, 1),
       offsets = (pid_m * BLOCK_M, pid_n * BLOCK_N),
       block_shape = (BLOCK_M, BLOCK_N),
       order = (1, 0)
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape = (n_elements, n_elements),
        strides = (n_elements, 1),
        offsets = (pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape = (BLOCK_M, BLOCK_N),
        order = (1, 0)
    )

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(n_elements, n_elements),
        strides=(n_elements, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )

    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    b = tl.load(b_block_ptr, boundary_check=(0, 1))
    c = a + b
    tl.store(c_block_ptr, c, boundary_check=(0, 1))

def solve_triton_2d(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    grid = lambda meta: (
        (triton.cdiv(N, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    )
    matrix_add_kernel_2d[grid](a, b, c, N)

def benchmark(func, A, B, C, N, warmup=10, repeat=100):
    # Warmup
    for _ in range(warmup):
        func(A, B, C, N)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(repeat):
        func(A, B, C, N)
    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / repeat * 10000
    print(f"Average time: {avg_time_ms:.6f} ms.")

def verify_results(C_triton_naive, C_triton_1d, C_triton_2d):
    if torch.allclose(C_triton_naive, C_triton_1d):
        print("Triton 1D kernel results match Triton naive kernel.")
    else:
        print("Triton 1D kernel results do NOT match Triton naive kernel.")

    if torch.allclose(C_triton_naive, C_triton_2d):
        print("Triton 2D kernel results match Triton naive kernel.")
    else:
        print("Triton 2D kernel results do NOT match Triton naive kernel.")

def main():
    device = torch.device('cuda')

    N = 1 << 12 # 4096
    A = torch.randn((N, N), device=device, dtype=torch.float32)
    B = torch.randn((N, N), device=device, dtype=torch.float32)
    C_triton_naive = torch.empty_like(A)
    C_triton_1d = torch.empty_like(A)
    C_triton_2d = torch.empty_like(A)

    solve_triton_naive(A, B, C_triton_naive, N)
    solve_triton_1d(A, B, C_triton_1d, N)
    solve_triton_2d(A, B, C_triton_2d, N)
    verify_results(C_triton_naive, C_triton_1d, C_triton_2d)

    # 性能测试
    print("\n开始性能测试 (预热 10 次，计时 100 次取平均)...\n")

    time_pytorch = benchmark(solve_triton_naive, A, B, C_triton_naive, N)
    time_triton_1d = benchmark(solve_triton_1d, A, B, C_triton_1d, N)
    time_triton_2d = benchmark(solve_triton_2d, A, B, C_triton_2d, N)

    # 输出结果
    print(f"Triton 直接加法:      {time_pytorch:.4f} ms")
    print(f"Triton 1D 向量化:      {time_triton_1d:.4f} ms")
    print(f"Triton 2D 块指针:      {time_triton_2d:.4f} ms")

    # 计算加速比
    baseline = time_pytorch
    print(f"\n相对于 Naive 的加速比:")
    print(f"  Triton 1D: {baseline / time_triton_1d:.2f}x")
    print(f"  Triton 2D: {baseline / time_triton_2d:.2f}x")

    # 计算内存带宽
    bytes_per_element = 4  # float32
    total_bytes = 3 * N * N * bytes_per_element  # A读 + B读 + C写
    bw_pytorch = total_bytes / (time_pytorch / 1000) / 1e9
    bw_triton_1d = total_bytes / (time_triton_1d / 1000) / 1e9
    bw_triton_2d = total_bytes / (time_triton_2d / 1000) / 1e9

    print(f"\n估算内存带宽 (GB/s):")
    print(f"  Triton Naive:  {bw_pytorch:.2f} GB/s")
    print(f"  Triton 1D: {bw_triton_1d:.2f} GB/s")
    print(f"  Triton 2D: {bw_triton_2d:.2f} GB/s")

if __name__ == "__main__":
    main()