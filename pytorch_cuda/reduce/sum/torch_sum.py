import torch

N = 1 << 26

x = torch.rand(N, device='cuda')

# 正确性校验：double 参考累加（同 cuda_sum.cu 的 verify 口径）
ref = x.double().sum().item()
out = x.sum().item()
assert abs(out - ref) / abs(ref) < 1e-3, f"sum failed: {out} vs {ref}"
print(f"correctness OK: N={N}")

_ = x.sum()  # 预热


def benchmark(func, *args, n_warmup=5, n_repeat=20):
    for _ in range(n_warmup):
        func(*args)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_repeat):
        func(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_repeat


ms = benchmark(lambda: x.sum())

# sum 只读一次 x，有效带宽 = N * 4 字节
bytes_total = N * 4
bw = (bytes_total / 1e9) / (ms / 1000.0)

print(f"PyTorch sum: {ms:.3f} ms, Bandwidth: {bw:.3f} GB/s")
