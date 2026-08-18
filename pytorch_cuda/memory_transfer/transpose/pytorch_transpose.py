import torch

rows, cols = 8192, 8192

x = torch.randn(rows, cols, device='cuda')

# 正确性校验：搬运后的连续结果应等于视图转置
y = x.t().contiguous()
assert torch.allclose(y, x.t()), "transpose copy failed"
print(f"correctness OK: {rows}x{cols} -> {cols}x{rows}")


def benchmark(fn, n_warmup=5, n_repeat=20):
    for _ in range(n_warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_repeat


# x.t() 只是视图（不搬数据），不能直接用来测带宽；
# 只有 .contiguous() 才会真正做一次转置内存搬运
ms = benchmark(lambda: x.t().contiguous())

# 转置搬运：读 x + 写 y = 2 * rows * cols * 4 字节
bytes_total = 2 * rows * cols * 4
bw = (bytes_total / 1e9) / (ms / 1000.0)

print(f"PyTorch Transpose (x.t().contiguous()): {ms:.3f} ms, Bandwidth: {bw:.3f} GB/s")