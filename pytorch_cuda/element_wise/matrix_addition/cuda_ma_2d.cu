#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==================== 两种 kernel ====================

// 版本1：naive 2D，每线程负责一个元素
__global__ void add_2d(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

// 版本2：2D grid-stride，固定线程数循环遍历（标量访存）
__global__ void add_2d_grid_stride(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_y = blockDim.y * gridDim.y;
    int stride_x = blockDim.x * gridDim.x;
    for (int r = row; r < M; r += stride_y) {
        for (int c = col; c < N; c += stride_x) {
            int idx = r * N + c;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// ==================== 启动封装 ====================

void launch_2d(const float* a, const float* b, float* c, int M, int N) {
    dim3 block(32, 8);   // 256 线程
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    add_2d<<<grid, block>>>(a, b, c, M, N);
}

void launch_2d_grid_stride(const float* a, const float* b, float* c, int M, int N) {
    dim3 block(32, 8);
    dim3 grid(min((N + block.x - 1) / block.x, 128), min((M + block.y - 1) / block.y, 128));
    add_2d_grid_stride<<<grid, block>>>(a, b, c, M, N);
}

// ==================== 正确性校验 ====================

int verify(void (*launch)(const float*, const float*, float*, int, int),
           const char* name) {
    int Ms[] = {1, 3, 100, 1000, 4096};   // 含非 4 倍数、非 256 倍数
    int Ns[] = {1, 5, 100, 999, 100};
    int n_cases = sizeof(Ms) / sizeof(Ms[0]);
    int ok = 1;

    for (int t = 0; t < n_cases; t++) {
        int M = Ms[t], N = Ns[t];
        int total = M * N;
        size_t bytes = (size_t)total * sizeof(float);

        float* h_a = (float*)malloc(bytes);
        float* h_b = (float*)malloc(bytes);
        float* h_c = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);
        for (int i = 0; i < total; i++) {
            h_a[i] = (float)rand() / RAND_MAX;
            h_b[i] = (float)rand() / RAND_MAX;
            h_ref[i] = h_a[i] + h_b[i];
        }

        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        launch(d_a, d_b, d_c, M, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

        double max_diff = 0.0;
        for (int i = 0; i < total; i++) {
            double diff = fabs((double)h_c[i] - (double)h_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff > 1e-6) {
            printf("%-24s  %dx%d  FAIL  max_diff=%.3e\n", name, M, N, max_diff);
            ok = 0;
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
        free(h_ref);
    }
    if (ok) printf("%-24s  ALL PASS (5 种形状)\n", name);
    return ok;
}

// ==================== 计时 + 带宽 ====================

void benchmark(void (*launch)(const float*, const float*, float*, int, int),
               const float* a, const float* b, float* c, int M, int N,
               const char* name) {
    int n_warmup = 5, n_repeat = 20;
    for (int i = 0; i < n_warmup; i++) launch(a, b, c, M, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_repeat; i++) launch(a, b, c, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_repeat;

    // C = A + B：读 A + 读 B + 写 C = 3 * M * N * 4 字节
    float gb = 3.0f * M * N * sizeof(float) / 1e9f;
    printf("%-24s  %8.3f ms   %8.1f GB/s\n", name, ms, gb / (ms / 1000.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ==================== leetgpu 风格接口 ====================

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    add_2d<<<grid, block>>>(A, B, C, N, N);
    cudaDeviceSynchronize();
}

// ==================== main ====================

int main() {
    // ==================== 正确性校验 ====================
    int pass = 1;
    pass &= verify(launch_2d, "add_2d");
    pass &= verify(launch_2d_grid_stride, "add_2d_grid_stride");
    printf("\n%s\n\n", pass ? "正确性校验：ALL PASS" : "正确性校验：SOME FAILED");
    if (!pass) return 1;

    // ==================== 性能测试 ====================
    int M = 8192, N = 8192;  // 2^26 个元素，与 vector_add 同量级
    size_t bytes = (size_t)M * N * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < M * N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    printf("M = %d, N = %d (%.2f GB per matrix)\n\n", M, N, bytes / 1e9f);
    benchmark(launch_2d, d_a, d_b, d_c, M, N, "add_2d");
    benchmark(launch_2d_grid_stride, d_a, d_b, d_c, M, N, "add_2d_grid_stride");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    return 0;
}