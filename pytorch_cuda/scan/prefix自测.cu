#include<cuda_runtime.h>

// 每个block的元素数量
constexpr int TILE = 1024;
// 每个block的线程数量
constexpr int THREADS = TILE / 2;

constexpr int LOG_NUM_BANK = 5;
// 动态共享内存的大小
constexpr int PADDED_TILE = TILE + (TILE >> LOG_NUM_BANK);

__device__ inline int pad(int i) {
    return i + (i >> LOG_NUM_BANK);
}

__global__ void scan_tile(float* input, float* output, float* totals, int N) {
    extern __shared__ float smem[];
    // la和lb是为了获取共享内存的索引
    int la = threadIdx.x;
    int lb = threadIdx.x + THREADS;
    // 注意一点，ga和gb的目的是获得数据N大小的全局索引，即input[ga]
    // 因此要blockIdx.x * TLLE, 而不是blockIdx.x * blockDim.x
    int ga = threadIdx.x + blockIdx.x * TILE;
    int gb = threadIdx.x + THREADS + blockIdx.x * TILE;
    smem[pad(la)] = (ga < N) ? input[ga] : 0.0f;
    smem[pad(lb)] = (gb < N) ? input[gb] : 0.0f;
    int offset = 1;
    for (int i = TILE >> 1; i > 0; i >>= 1) {
        __syncthreads();
        if (threadIdx.x < i) {
            int left_idx = offset * (2 * threadIdx.x + 1) - 1;
            int right_idx = offset * (2 * threadIdx.x + 2) - 1;
            smem[pad(left_idx)] += smem[pad(right_idx)];
        }
        offset <<= 1;
    }

    __syncthreads();
    if (threaIdx.x == 0) {
        totals[blockIdx.x] = smem[pad(TILE - 1)];
        smem[pad(TILE - 1)] = 0.0f;
    }

    for (int i = 1; i < TILE; i <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (threadIdx.x < i) {
            int left_idx = offset * (2 * threadIdx.x + 1) - 1;
            int right_idx = offset * (2 * threadIdx.x + 2) - 1;
            int pad_left = pad(left_idx);
            int pad_right = pad(right_idx);
            float temp = smem[pad_left];
            smem[pad_left] = smem[pad_right];
            smem[pad_right] += temp;
        }
    }
    __syncthreads();

    if (ga < N) output[ga] = smem[pad(la)];
    if (gb < N) output[gb] = smem[pad(lb)];
}

__global__ void carry_add(float* totals, float* output, int N) {
    int gi = threadIdx.x + blockIdx.x * TILE;
    int gj = threadIdx.x + THREADS + blockIdx.x * TILE;
    float c = totals[blockIdx.x];

    if (gi < N) output[gi] += c;
    if (gj < N) output[gj] += c;
}

void scan_recursive(float* data, int N) {
    if (N <= TILE) {
        float* dummy_total;
        cudaMalloc(&dummy_total, sizeof(float));
        scan_tile<<<1, THREADS, PADDED_TILE * sizeof(float)>>>(data, data, dummy_total, N);
        return;
    }

    int tiles = (N + TILE - 1) / TILE;
    float* totals;
    cudaMalloc(&totals, sizeof(float) * tiles);

    scan_tile<<<tiles, THREADS, PADDED_TILE * sizeof(float)>>>(data, data, totals, N);
    scan_recursive(totals, tiles);
    add_carry<<<tiles, THREADS>>>(totals, data, N);
    cudaFree(totals);
}

extern "C" void solve(const float* input, float* output, int N) {
    cudaMemcpy(output, input, sizeof(float) * N, cudaMemcpyDeviceToDevice);
    scan_recursive(output, N);
}