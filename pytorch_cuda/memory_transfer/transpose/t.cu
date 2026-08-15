#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err),     \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

/**
 * 使用共享内存优化的矩阵转置内核
 * 
 * 算法思路：
 * 1. 每个 block 加载一个 tile（32x32）的数据到共享内存
 * 2. 读取时按行读取（coalesced global memory access）
 * 3. 写出时交换 threadIdx.x 和 threadIdx.y，实现转置
 * 4. 共享内存列维度 +1 padding，避免 bank conflict
 *
 * Bank Conflict 分析：
 *   读取阶段：tile[y][x]，不同线程访问不同 x → 不同 bank，无冲突
 *   写出阶段：tile[x][y]，不同线程访问不同 y → 不同 bank，无冲突
 *   若无 +1 padding，tile[y][x] 在 x 相同时会冲突
 *
 * @param in   输入矩阵（行优先，大小 rows × cols）
 * @param out  输出矩阵（转置后，大小 cols × rows）
 * @param rows 输入矩阵行数
 * @param cols 输入矩阵列数
 */
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // 当前线程在全局矩阵中的行列坐标
    int myRow = blockIdx.y * blockDim.y + threadIdx.y;
    int myCol = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前 tile 的左上角在全局矩阵中的坐标
    int tileX = blockIdx.x * blockDim.x;
    int tileY = blockIdx.y * blockDim.y;

    // 共享内存，+1 padding 避免 bank conflict
    __shared__ float tile[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X + 1];

    // ---- 阶段1：从全局内存按行读入共享内存 ----
    // 线程 (threadIdx.y, threadIdx.x) 读取 in[myRow][myCol]
    int inIdx = myRow * cols + myCol;
    if (myRow < rows && myCol < cols)
        tile[threadIdx.y][threadIdx.x] = in[inIdx];

    // 同步，确保所有线程都已完成共享内存写入
    __syncthreads();

    // ---- 阶段2：从共享内存写出到全局内存（转置） ----
    // 转置：out[j][i] = in[i][j]
    // 写出的全局坐标是 (tileX + threadIdx.y, tileY + threadIdx.x)
    // 这里交换了 x 和 y，实现转置
    int outRow = tileX + threadIdx.y;  // 转置后的行 = 原来的列块 + threadIdx.y
    int outCol = tileY + threadIdx.x;  // 转置后的列 = 原来的行块 + threadIdx.x
    // 为了保证outIdx在全局矩阵里是可以内存合并的，这里用outRow*rows+outCol
    // 而不是outCol*cols+outRow
    // 这样构造出来的转置矩阵是列优先的
    int outIdx = outRow * rows + outCol;
    if (outRow < cols && outCol < rows)
        out[outIdx] = tile[threadIdx.x][threadIdx.y];
}

/**
 * CPU 纯 C++ 矩阵转置
 * 直接用双重循环按索引完成，无任何优化，作为性能基准
 */
void transposeCPU(const float* in, float* out, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out[j * rows + i] = in[i * cols + j];
        }
    }
}

/**
 * CPU 端验证转置结果
 * @return 最大绝对误差
 */
double verifyTranspose(const float* h_in, float* h_out, int rows, int cols) {
    double maxErr = 0.0;
    int errCount = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double expected = (double)h_in[i * cols + j];
            double actual   = (double)h_out[j * rows + i];
            double err = fabs(actual - expected);
            if (err > maxErr) maxErr = err;
            if (err > 1e-5 && errCount < 5) {
                printf("  MISMATCH at out[%d][%d]: expected %.6f, got %.6f\n",
                       j, i, expected, actual);
                errCount++;
            }
        }
    }
    return maxErr;
}

int main(int argc, char* argv[]) {
    int rows = 1024;
    int cols = 1024;
    size_t bytes = rows * cols * sizeof(float);

    printf("=== 矩阵转置测试 (共享内存 + bank conflict 优化) ===\n");
    printf("矩阵大小: %d x %d (%d 元素)\n", rows, cols, rows * cols);

    // 分配主机内存并用随机值初始化
    float* h_in    = (float*)malloc(bytes);
    float* h_out   = (float*)malloc(bytes);
    float* h_out_cpu = (float*)malloc(bytes);
    for (int i = 0; i < rows * cols; ++i) {
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // ========== CPU 转置 ==========
    clock_t cpuStart = clock();
    transposeCPU(h_in, h_out_cpu, rows, cols);
    clock_t cpuEnd = clock();
    double cpuMs = (double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC * 1000.0;
    printf("\n--- CPU 转置 (纯 C++ 双重循环) ---\n");
    printf("耗时: %.3f ms\n", cpuMs);
    printf("带宽: %.2f GB/s\n", 2.0 * bytes / (cpuMs * 1e6));

    // ========== GPU 转置 ==========

    // 分配设备内存
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));

    // 复制输入到设备
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 配置 kernel 启动参数
    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    printf("\n--- GPU 转置 (共享内存 + padding) ---\n");
    printf("Block: %dx%d, Grid: %dx%d\n", block.x, block.y, grid.x, grid.y);

    // 用 CUDA Event 计时
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    transpose<<<grid, block>>>(d_in, d_out, rows, cols);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 复制结果回主机
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 验证结果
    double maxErr = verifyTranspose(h_in, h_out, rows, cols);

    printf("内核耗时: %.3f ms\n", ms);
    printf("带宽: %.2f GB/s\n", 2.0 * bytes / (ms * 1e6));
    printf("最大误差: %.6e\n", maxErr);
    printf("%s\n", maxErr < 1e-5 ? "PASSED" : "FAILED");

    // ========== 对比 ==========
    printf("\n=== CPU vs GPU 对比 ===\n");
    printf("CPU 耗时: %.3f ms\n", cpuMs);
    printf("GPU 耗时: %.3f ms\n", ms);
    printf("加速比: %.1fx\n", cpuMs / ms);

    // 释放资源
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_out_cpu);
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return 0;
}
