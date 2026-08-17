#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ==================== 三种 kernel ====================

// 版本1：naive，每线程负责一个元素
__global__ void add_naive(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// 版本2：grid-stride 标量，固定线程数循环遍历
__global__ void add_grid_stride(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// 版本3：grid-stride + float4 向量化，一次读写 4 个 float
__global__ void add_grid_stride_vec4(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    int n4 = n / 4;
    for (int i = idx; i < n4; i += stride) {
        float4 x = a4[i], y = b4[i];
        c4[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
    }

    // 尾部不足 4 个的元素，标量兜底
    int remain = n4 * 4;
    for (int i = remain + idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ==================== 启动封装 ====================

void launch_naive(const float* a, const float* b, float* c, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_naive<<<grid, block>>>(a, b, c, n);
}

void launch_grid_stride(const float* a, const float* b, float* c, int n) {
    int block = 256;
    int grid = min((n + block - 1) / block, 1024);
    add_grid_stride<<<grid, block>>>(a, b, c, n);
}

void launch_grid_stride_vec4(const float* a, const float* b, float* c, int n) {
    int block = 256;
    int grid = min((n + block - 1) / block, 1024);
    add_grid_stride_vec4<<<grid, block>>>(a, b, c, n);
}

// ==================== 计时 + 带宽 ====================

void benchmark(void (*launch)(const float*, const float*, float*, int),
               const float* a, const float* b, float* c, int n,
               const char* name) {
    int n_warmup = 5, n_repeat = 20;
    for (int i = 0; i < n_warmup; i++) launch(a, b, c, n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_repeat; i++) launch(a, b, c, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_repeat;

    // C = A + B：读 A + 读 B + 写 C = 3 * n * 4 字节
    float gb = 3.0f * n * sizeof(float) / 1e9f;
    printf("%-24s  %8.3f ms   %8.1f GB/s\n", name, ms, gb / (ms / 1000.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ==================== main ====================

int main() {
    int N = 1 << 26;  // 64 M，远超 L2，主要测的是 HBM 带宽
    size_t bytes = (size_t)N * sizeof(float);

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    printf("N = %d (%.2f GB per array)\n\n", N, bytes / 1e9f);
    benchmark(launch_naive, d_a, d_b, d_c, N, "add_naive");
    benchmark(launch_grid_stride, d_a, d_b, d_c, N, "add_grid_stride");
    benchmark(launch_grid_stride_vec4, d_a, d_b, d_c, N, "add_grid_stride_vec4");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    return 0;
}