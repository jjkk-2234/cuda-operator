#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

namespace cg = cooperative_groups;

// ==================== Kernel 实现 ====================

__device__ __forceinline__ float warpReduceSum(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE>
__global__ void reduceModern(const float* __restrict__ in,
                             float* __restrict__ out,
                             int N) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int globalIdx = blockIdx.x * BLOCK_SIZE + tid;

    // Grid-Stride Loop
    float threadSum = 0.0f;
    for (int idx = globalIdx; idx < N; idx += BLOCK_SIZE * gridDim.x) {
        threadSum += in[idx];
    }

    // Warp级归约
    float warpSum = warpReduceSum(threadSum);

    // Block级归约
    if ((tid % warpSize) == 0) {
        smem[tid / warpSize] = warpSum;
    }
    __syncthreads();

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    if (tid < NUM_WARPS) {
        warpSum = smem[tid];
    } else {
        warpSum = 0.0f;
    }

    if (tid < 32) {
        warpSum = warpReduceSum(warpSum);
    }

    if (tid == 0) {
        atomicAdd(out, warpSum);
    }
}

// ==================== 工具函数 ====================

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// GPU高精度计时封装
struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    float end() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// CPU参考实现（使用double避免浮点累积误差过大）
double cpuReduce(const float* data, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) sum += (double)data[i];
    return sum;
}

bool verifyResult(double gpuResult, double cpuResult, float relTol = 1e-4f) {
    double diff = fabs(gpuResult - cpuResult);
    double denom = fmax(fabs(cpuResult), 1.0);
    bool pass = (diff / denom) < relTol;
    printf("  CPU Reference : %.6f\n", cpuResult);
    printf("  GPU Result    : %.6f\n", gpuResult);
    printf("  Relative Error: %.2e  [%s]\n", diff / denom, pass ? "PASS ✓" : "FAIL ✗");
    return pass;
}

// ==================== 主函数 ====================

int main(int argc, char** argv) {
    // 打印设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Device: %s ===\n", prop.name);
    printf("  SM Count     : %d\n", prop.multiProcessorCount);
    printf("  Mem Bandwidth: %.1f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("\n");

    // 测试不同规模
    const int sizes[] = {1 << 20, 1 << 24, 1 << 28}; // 1M, 16M, 256M
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARMUP_ITERS = 10;
    constexpr int BENCH_ITERS = 100;

    for (int s = 0; s < numSizes; s++) {
        int N = sizes[s];
        size_t bytes = N * sizeof(float);
        printf("--- N = %d (%.1f MB) ---\n", N, bytes / (1024.0 * 1024.0));

        // 分配 & 初始化
        float* h_in = new float[N];
        for (int i = 0; i < N; i++) h_in[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        double cpuResult = cpuReduce(h_in, N);
        size_t smemSize = (BLOCK_SIZE / 32) * sizeof(float);
        int numBlocks = std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                                 prop.multiProcessorCount * 4);

        GpuTimer timer;
        float customMs, cubMs;

        // ===== Warmup =====
        for (int i = 0; i < WARMUP_ITERS; i++) {
            CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
            reduceModern<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, smemSize>>>(d_in, d_out, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // ===== Benchmark: Custom Kernel =====
        timer.begin();
        for (int i = 0; i < BENCH_ITERS; i++) {
            CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
            reduceModern<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, smemSize>>>(d_in, d_out, N);
        }
        customMs = timer.end() / BENCH_ITERS;

        // 验证正确性
        float h_out;
        CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  [Custom] ");
        verifyResult((double)h_out, cpuResult);
        printf("  Avg Time: %.4f ms | BW: %.1f GB/s\n",
               customMs, bytes / customMs / 1e6);

        // ===== Benchmark: CUB (作为性能上限参考) =====
        void* d_temp = nullptr;
        size_t tempBytes = 0;
        cub::DeviceReduce::Sum(d_temp, tempBytes, d_in, d_out, N);
        CUDA_CHECK(cudaMalloc(&d_temp, tempBytes));

        // Warmup CUB
        for (int i = 0; i < WARMUP_ITERS; i++)
            cub::DeviceReduce::Sum(d_temp, tempBytes, d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.begin();
        for (int i = 0; i < BENCH_ITERS; i++)
            cub::DeviceReduce::Sum(d_temp, tempBytes, d_in, d_out, N);
        cubMs = timer.end() / BENCH_ITERS;

        printf("  [CUB]    Avg Time: %.4f ms | BW: %.1f GB/s\n",
               cubMs, bytes / cubMs / 1e6);
        printf("  Custom/CUB Ratio: %.2fx\n\n", customMs / cubMs);

        // Cleanup
        CUDA_CHECK(cudaFree(d_temp));
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        delete[] h_in;
    }

    return 0;
}