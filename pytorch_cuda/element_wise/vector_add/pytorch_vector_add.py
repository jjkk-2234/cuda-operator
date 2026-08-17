import torch

N = 1 << 26

a = torch.randn(N, device='cuda')
b = torch.randn(N, device='cuda')

# 预热
_ = a + b

def benchmark_add(func, *args, name="Add", n_warmup=5, n_repeat=20):
    # 预热
    for _ in range(n_warmup):
        func(*args)
        
    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_repeat):
        func(*args)
    end.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start.elapsed_time(end) / n_repeat
    return elapsed_time_ms

ms = benchmark_add(lambda x, y: x + y, a, b, name="PyTorch Add")

# 计算有效带宽
bytes_total = 3 * N * 4
bw = (bytes_total / 1e9) / (ms / 1000.0)

print(f"PyTorch Add: {ms:.3f} ms, Bandwidth: {bw:.3f} GB/s")