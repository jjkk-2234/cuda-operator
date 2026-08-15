#include <cuda_runtime.h>

constexpr int NUM_BANKS = 32;
constexpr int LOG_NUM_BANKS = 5;

__device__ inline int pad(int i) {
    return i + (i >> LOG_NUM_BANKS);
}

constexpr int TILE = 1024;
constexpr int THREADS = TILE / 2;
constexpr int PADDED_TILE = TILE + (TILE >> LOG_NUM_BANKS);

// ============ 块内 Blelloch INCLUSIVE scan + 存块总和 ============
// 上下扫仍是 exclusive 的，最后写回 + xa 补上当前元素，变成 inclusive。
__global__ void scan_tile(const float* in, float* out, float* totals, int n) {
    extern __shared__ float temp[];

    int thid = threadIdx.x;
    int offset = 1;

    int ai = thid + blockIdx.x * TILE;
    int bi = thid + (TILE >> 1) + blockIdx.x * TILE;
    int a  = thid;
    int b  = thid + (TILE >> 1);

    // 原值留在寄存器，inclusive 写回时直接用，避免二次读全局
    float xa = (ai < n) ? in[ai] : 0.0f;
    float xb = (bi < n) ? in[bi] : 0.0f;
    temp[pad(a)] = xa;
    temp[pad(b)] = xb;

    // upsweep
    for (int d = TILE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int left_idx  = offset * (2 * thid + 1) - 1;
            int right_idx = offset * (2 * thid + 2) - 1;
            temp[pad(right_idx)] += temp[pad(left_idx)];
        }
        offset <<= 1;
    }

    __syncthreads();
    if (thid == 0) {
        totals[blockIdx.x] = temp[pad(TILE - 1)];  // 块总和（upsweep 根节点值）
        temp[pad(TILE - 1)] = 0.0f;                 // 根清零 → exclusive
    }

    // downsweep
    for (int d = 1; d < TILE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int left_idx  = offset * (2 * thid + 1) - 1;
            int right_idx = offset * (2 * thid + 2) - 1;
            int pad_left  = pad(left_idx);
            int pad_right = pad(right_idx);
            float t = temp[pad_left];
            temp[pad_left]  = temp[pad_right];
            temp[pad_right] += t;
        }
    }
    __syncthreads();

    // 此刻 temp 里是块内 exclusive scan，加寄存器里的原值 → 块内 inclusive scan
    if (ai < n) out[ai] = temp[pad(a)] + xa;
    if (bi < n) out[bi] = temp[pad(b)] + xb;
}

// ============ 把每块 carry 加回块内所有元素 ============
// totals 是 INCLUSIVE 前缀（含本块），carry 需要"前 k-1 块之和"，
// 所以错位读前一块的值；块 0 的 carry 为 0。
__global__ void add_carry(const float* totals, float* out, int n) {
    int gi = blockIdx.x * TILE + threadIdx.x;
    int gj = threadIdx.x + TILE >> 1 + blockIdx.x * TILE;
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
        scan_tile<<<1, THREADS, PADDED_TILE * sizeof(float)>>>(data, data, dummy, n);
        cudaFree(dummy);
        return;
    }

    int tiles = (n + TILE - 1) / TILE;
    float* totals;
    cudaMalloc(&totals, tiles * sizeof(float));

    scan_tile<<<tiles, THREADS, PADDED_TILE * sizeof(float)>>>(data, data, totals, n);
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
