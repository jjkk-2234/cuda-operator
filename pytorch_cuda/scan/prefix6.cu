#include <cuda_runtime.h>

constexpr int TILE = 1024;
constexpr int THREADS = TILE / 2;   // 512，每线程 2 元素
constexpr int WARPS = THREADS / 32; // 16
constexpr int LANES = 32;

// ============ 块内 warp-shuffle INCLUSIVE scan + 存块总和 ============
// 两级 scan：线程内 2 元素 → warp scan（shuffle）→ warp0 扫 warp 和。
// 全程只有 2 次 __syncthreads()，无共享内存 bank 冲突。
__global__ void scan_tile(const float* in, float* out, float* totals, int n) {
    // 设定有16个warp，每个线程处理2个元素
    __shared__ float warp_sums[WARPS];
    
    int thid  = threadIdx.x;
    // warp代表第几个warp
    int warp  = thid >> 5;
    // lane代表第几个线程
    int lane  = thid & 31;

    int ai = blockIdx.x * TILE + thid * 2;   // 元素 2t（连续）
    int bi = ai + 1;                          // 元素 2t+1

    float xa = (ai < n) ? in[ai] : 0.0f;
    float xb = (bi < n) ? in[bi] : 0.0f;

    // 线程内 2 元素：pair 是线程负载的总和，pre 是 xa 之前的线程内前缀
    float pair = xa + xb;

    // 一级：warp 内 inclusive scan（5 次 shuffle，无 barrier），使用了Hillis-Steele算法
    float sum = pair;
    #pragma unroll
    for (int d = 1; d < LANES; d <<= 1) {
        float t = __shfl_up_sync(0xffffffff, sum, d);
        if (lane >= d) sum += t;
    }
    // sum = 本 lane 及之前所有 lane 的 pair 之和（inclusive）
    // 本线程线程内 exclusive 前缀 pre = sum - pair

    // warp 和存共享，warp0 扫 16 个 warp 和
    if (lane == LANES - 1) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        float ws = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int d = 1; d < LANES; d <<= 1) {
            float t = __shfl_up_sync(0xffffffff, ws, d);
            if (lane >= d) ws += t;
        }
        // ws = 前 (lane+1) 个 warp 之和（就是inclusive）；lane==15 即块总和
        if (lane == WARPS - 1) totals[blockIdx.x] = ws;  // 块总和，先存再覆盖
        float carry = ws - ((lane < WARPS) ? warp_sums[lane] : 0.0f); // exclusive 前缀
        if (lane < WARPS) warp_sums[lane] = carry;  // 覆盖为"本 warp 前所有 warp 之和", 就是exclusive前缀
    }
    __syncthreads();

    // 二级：本线程 carry = 本 warp 之前所有 warp 的总和
    float cw   = warp_sums[warp];
    float pre  = sum - pair;   // 本 warp 中本线程之前的元素总和
    // pre 是 exclusive，但它同时含了本线程的 xa（pre = warp 前缀 - pair = 前 t 个线程的全部）
    // 所以 xa 的块内 inclusive = cw + pre + xa，xb 的 = cw + pre + xa + xb
    if (ai < n) out[ai] = cw + pre + xa;
    if (bi < n) out[bi] = cw + pre + pair;
}

// ============ 把每块 carry 加回块内所有元素 ============
// totals 是 INCLUSIVE 前缀（含本块），carry 需要"前 k-1 块之和"，
// 所以错位读前一块的值；块 0 的 carry 为 0。
__global__ void add_carry(const float* totals, float* out, int n) {
    int gi = blockIdx.x * TILE + threadIdx.x;
    int gj = gi + TILE / 2;
    float c = (blockIdx.x == 0) ? 0.0f : totals[blockIdx.x - 1];

    if (gi < n) out[gi] += c;
    if (gj < n) out[gj] += c;
}

// ============ 递归：把"块总和数组"扫成 INCLUSIVE 前缀 ============
// scan_tile 本身就是 inclusive，totals 层直接用同一内核。
// carry 靠 add_carry 错位读（totals[k-1]）得到"前 k-1 块之和"。
void scan_recursive(float* data, int n) {
    if (n <= TILE) {
        float* dummy;
        cudaMalloc(&dummy, sizeof(float));
        scan_tile<<<1, THREADS>>>(data, data, dummy, n);
        cudaFree(dummy);
        return;
    }

    int tiles = (n + TILE - 1) / TILE;
    float* totals;
    cudaMalloc(&totals, tiles * sizeof(float));

    scan_tile<<<tiles, THREADS>>>(data, data, totals, n);
    scan_recursive(totals, tiles);
    add_carry<<<tiles, THREADS>>>(totals, data, n);

    cudaFree(totals);
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N == 0) return;

    // input 是 const，不能就地；先拷到 output，再对 output 就地递归
    cudaMemcpy(output, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    scan_recursive(output, N);
}