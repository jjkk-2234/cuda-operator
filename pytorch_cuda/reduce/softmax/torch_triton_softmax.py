import triton
import triton.language as tl
import torch
import time

# ==================== PyTorch 参考实现 ====================

def torch_softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)

# ==================== Triton 实现 ====================
@triton.jit
def kernel_softmax_fuse(
    x_ptr, x_row_stride,
    y_ptr, y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float('inf'))
    x = tl.exp(x - tl.max(x, axis=0))
    eps = float(1e-9)
    x /= tl.maximum(tl.sum(x), eps)
    tl.store(y_ptr + idx, x, mask = idx < n_cols)

def triton_softmax_fuse(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_fuse[[n_rows]](x, x.stride(0), y, y.stride(0), n_cols, BLOCK_SIZE=triton.next_power_of_2(n_cols), num_warps=32)
    return y

@triton.jit
def kernel_softmax_tile(
    x_ptr, x_row_stride,
    y_ptr, y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float('inf'))
        mm = tl.maximum(mm, x)
    mm = tl.max(mm)
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm)
            ss += x
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm)
            ss += x
    ss = tl.sum(ss)
    eps = float(1e-9)
    ss = tl.maximum(ss, eps)
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
        x = tl.exp(x - mm) / ss
        tl.store(y_ptr + idx, x, mask = idx < n_cols)

def triton_softmax_tile(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_tile[[n_rows]](x, x.stride(0), y, y.stride(0), n_cols, BLOCK_SIZE=2**14, CACHE_OPT=cache_opt, num_warps=32)
    return y

@triton.jit
def kernel_softmax_online(
    x_ptr, x_row_stride,
    y_ptr, y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
        mm_new = tl.maximum(mm, x)
        if i:
            ss *= tl.exp(mm - mm_new)
        x = tl.exp(x - mm_new)
        ss += tl.where(idx < n_cols, x, 0.0)
        mm = mm_new
    mm_new = tl.max(mm)
    ss *= tl.exp(mm - mm_new)
    ss = tl.sum(ss)
    mm = mm_new
    eps = float(1e-9)
    ss = tl.maximum(ss, eps)
    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask = idx < n_cols)
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask = idx < n_cols)

def triton_softmax_online(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_online[[n_rows]](
        x, x.stride(0),
        y, y.stride(0),
        n_cols,
        BLOCK_SIZE=2**12,
        CACHE_OPT=cache_opt,
        num_warps=32,
    )
    return y

# ==================== 正确性校验 ====================

def verify():
    cases = [(100, 100), (1000, 1000), (100, 4096)]
    ok = True
    for M, N in cases:
        x = torch.randn(M, N, device='cuda')
        ref = torch_softmax(x)
        y1 = triton_softmax_fuse(x)
        y2 = triton_softmax_tile(x)
        y3 = triton_softmax_online(x)
        d1 = (y1 - ref).abs().max().item()
        d2 = (y2 - ref).abs().max().item()
        d3 = (y3 - ref).abs().max().item()
        if d1 > 1e-5 or d2 > 1e-5 or d3 > 1e-5:
            print(f"softmax  {M}x{N}  FAIL  max_diff={d1:.3e}, {d2:.3e}, {d3:.3e}")
            ok = False
    if ok:
        print("softmax  ALL PASS (3 种形状)")
    return ok

# ==================== 性能测试 ====================

def benchmark():
    M, N = 8192, 8192
    x = torch.randn(M, N, device='cuda')
    n_warmup, n_repeat = 5, 20
    
    def _bench(fn, name):
        for _ in range(n_warmup): fn(x)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_repeat): fn(x)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / n_repeat
        gb = 2.0 * M * N * 4 / 1e9
        print(f"{name:32s}  {ms:8.3f} ms   {gb / (ms / 1000):8.1f} GB/s")
    
    print(f"M = {M}, N = {N} ({M*N*4/1e9:.2f} GB)\n")
    _bench(lambda x: torch_softmax(x), "torch softmax")
    _bench(lambda x: triton_softmax_fuse(x), "triton softmax_fuse")
    _bench(lambda x: triton_softmax_tile(x), "triton softmax_tile")
    _bench(lambda x: triton_softmax_online(x), "triton softmax_online")

# ==================== main ====================

if __name__ == "__main__":
    print("正确性校验：")
    verify()
    print("\n性能测试：")
    benchmark()
