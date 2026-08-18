#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define TILE 32

// ==================== 两个 kernel ====================

// 版本1：naive 2D 转置，每线程负责一个元素
// 读端连续（合并），写端 out[col*rows+row] 跨行访问（不合并）
__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

// 版本2：共享内存 tile 转置，+1 padding 避免 bank conflict
// 读：按行连续读入 smem（合并）；写：交换 x/y 后写出（合并）
__global__ void transpose_smem(const float* __restrict__ in,
                               float* __restrict__ out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];

    int tileX = blockIdx.x * TILE;
    int tileY = blockIdx.y * TILE;
    int myRow = tileY + threadIdx.y;
    int myCol = tileX + threadIdx.x;

    // 阶段1：读入共享内存（按行，coalesced）
    if (myRow < rows && myCol < cols)
        tile[threadIdx.y][threadIdx.x] = in[myRow * cols + myCol];

    __syncthreads();

    // 阶段2：转置写出（交换 x/y）
    int outRow = tileX + threadIdx.y;  // 转置后行 = 原列块
    int outCol = tileY + threadIdx.x;  // 转置后列 = 原行块
    if (outRow < cols && outCol < rows)
        out[outRow * rows + outCol] = tile[threadIdx.x][threadIdx.y];
}

// ==================== 启动封装 ====================

void launch_naive(const float* in, float* out, int rows, int cols) {
    dim3 block(TILE, TILE);   // 1024 线程
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transpose_naive<<<grid, block>>>(in, out, rows, cols);
}

void launch_smem(const float* in, float* out, int rows, int cols) {
    dim3 block(TILE, TILE);   // 1024 线程，tile 32x32
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    transpose_smem<<<grid, block>>>(in, out, rows, cols);
}

// ==================== CPU 参考实现 ====================

void transpose_cpu(const float* in, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[j * rows + i] = in[i * cols + j];
}

// ==================== 正确性校验 ====================

int verify(void (*launch)(const float*, float*, int, int), const char* name) {
    int Ms[] = {1024, 512, 2048, 7, 100};    // 含非方阵、非 32 倍数
    int Ns[] = {1024, 2048, 512, 11, 999};
    int n_cases = sizeof(Ms) / sizeof(Ms[0]);
    int ok = 1;

    for (int t = 0; t < n_cases; t++) {
        int rows = Ms[t], cols = Ns[t];
        int total = rows * cols;
        size_t bytes = (size_t)total * sizeof(float);

        float* h_in = (float*)malloc(bytes);
        float* h_out = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);
        for (int i = 0; i < total; i++) h_in[i] = (float)rand() / RAND_MAX;
        transpose_cpu(h_in, h_ref, rows, cols);

        float *d_in, *d_out;
        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);
        cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

        launch(d_in, d_out, rows, cols);
        cudaDeviceSynchronize();

        cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

        double max_diff = 0.0;
        for (int i = 0; i < total; i++) {
            double diff = fabs((double)h_out[i] - (double)h_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff > 1e-6) {
            printf("%-24s  %dx%d  FAIL  max_diff=%.3e\n", name, rows, cols, max_diff);
            ok = 0;
        }

        cudaFree(d_in);
        cudaFree(d_out);
        free(h_in);
        free(h_out);
        free(h_ref);
    }
    if (ok) printf("%-24s  ALL PASS (5 种形状)\n", name);
    return ok;
}

// ==================== 计时 + 带宽 ====================

float benchmark(void (*launch)(const float*, float*, int, int),
                const float* in, float* out, int rows, int cols,
                const char* name) {
    int n_warmup = 5, n_repeat = 20;
    for (int i = 0; i < n_warmup; i++) launch(in, out, rows, cols);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_repeat; i++) launch(in, out, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_repeat;

    // 转置：读 in + 写 out = 2 * M * N * 4 字节
    float gb = 2.0f * rows * cols * sizeof(float) / 1e9f;
    printf("%-24s  %8.3f ms   %8.1f GB/s\n", name, ms, gb / (ms / 1000.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ==================== main ====================

int main() {
    // ==================== 正确性校验 ====================
    int pass = 1;
    pass &= verify(launch_naive, "transpose_naive");
    pass &= verify(launch_smem, "transpose_smem");
    printf("\n%s\n\n", pass ? "正确性校验：ALL PASS" : "正确性校验：SOME FAILED");
    if (!pass) return 1;

    // ==================== 性能测试 ====================
    int rows = 8192, cols = 8192;  // 2^26 个元素，与 vector_add / matrix_add 同量级
    size_t bytes = (size_t)rows * cols * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < rows * cols; i++) h_in[i] = (float)rand() / RAND_MAX;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    printf("rows = %d, cols = %d (%.2f GB per matrix)\n\n", rows, cols, bytes / 1e9f);
    float ms_naive = benchmark(launch_naive, d_in, d_out, rows, cols, "transpose_naive");
    float ms_smem  = benchmark(launch_smem,  d_in, d_out, rows, cols, "transpose_smem");

    // ==================== CPU 参考计时 ====================
    // 纯 C++ 双重循环，无缓存优化，仅作基准（8192x8192 会耗时数秒）
    float* h_out_cpu = (float*)malloc(bytes);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    transpose_cpu(h_in, h_out_cpu, rows, cols);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("transpose_cpu           %8.3f ms   %8.1f GB/s\n", cpu_ms, 2.0 * bytes / (cpu_ms * 1e6));

    // ==================== CPU vs GPU 对比 ====================
    printf("\n=== CPU vs GPU 对比 ===\n");
    printf("CPU 耗时: %.3f ms\n", cpu_ms);
    printf("GPU(smem) 耗时: %.3f ms\n", ms_smem);
    printf("加速比: %.1fx\n", cpu_ms / ms_smem);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out_cpu);
    return 0;
}