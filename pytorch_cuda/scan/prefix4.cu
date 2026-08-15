#include <cuda_runtime.h>

constexpr int NUM_BANKS = 32;
constexpr int LOG_NUM_BANKS = 5;

__device__ inline int pad(int i) {
    return i + (i >> LOG_NUM_BANKS);
}

constexpr int TILE = 1024;
constexpr int THREADS = TILE / 2;
constexpr int PADDED_TILE = TILE + (TILE >> LOG_NUM_BANKS);

// ============ Kernel 1：块内 Blelloch exclusive scan + 存块总和 ============
__global__ void scan_tile(const float* in, float* out, float* totals, int n) {
    extern __shared__ float temp[];

    int thid = threadIdx.x;
    int offset = 1;

    int ai = thid + blockIdx.x * TILE;
    int bi = thid + (TILE >> 1) + blockIdx.x * TILE;
    int a  = thid;
    int b  = thid + (TILE >> 1);

    temp[pad(a)] = (ai < n) ? in[ai] : 0.0f;
    temp[pad(b)] = (bi < n) ? in[bi] : 0.0f;

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
        totals[blockIdx.x] = temp[pad(TILE - 1)];  // 存块总和
        temp[pad(TILE - 1)] = 0.0f;                 // 根清零，为 exclusive 做准备
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

    // 每线程读写同一批地址（ai/bi），就地安全
    if (ai < n) out[ai] = temp[pad(a)];
    if (bi < n) out[bi] = temp[pad(b)];
}

// ============ Kernel 2：把每块 carry 加回块内所有元素 ============
__global__ void add_carry(const float* totals, float* out, int n) {
    int gi = blockIdx.x * TILE + threadIdx.x;
    int gj = threadIdx.x + TILE >> 1 + blockIdx.x * TILE;
    float c = totals[blockIdx.x];

    if (gi < n) out[gi] += c;
    if (gj < n) out[gj] += c;
}

// ============ 递归：totals 装得下就到底，否则再分层 ============
void scan_recursive(float* data, int n) {
    if (n <= TILE) {
        float* dummy;
        cudaMalloc(&dummy, sizeof(float));
        // 一层就够：单块扫完整数组，无需 totals
        scan_tile<<<1, THREADS, PADDED_TILE * sizeof(float)>>>(data, data, dummy, n);
        return;
    }
    // 分层，这里的层是什么？是根据TILE划分数据N得到的block数
    int tiles = (n + TILE - 1) / TILE;
    float* totals;
    cudaMalloc(&totals, tiles * sizeof(float));

    // 1. 每块扫自己那块，同时收集块总和到 totals
    scan_tile<<<tiles, THREADS, PADDED_TILE * sizeof(float)>>>(data, data, totals, n);
    // 2. 递归：把 totals 本身扫成 exclusive 前缀（即每块 carry）
    scan_recursive(totals, tiles);
    // 3. 每块把 carry 加回自己的块内 scan 结果
    add_carry<<<tiles, THREADS>>>(totals, data, n);

    cudaFree(totals);
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N == 0) return;

    // input 是 const，不能就地；先拷到 output，再对 output 就地递归
    cudaMemcpy(output, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    scan_recursive(output, N);
}