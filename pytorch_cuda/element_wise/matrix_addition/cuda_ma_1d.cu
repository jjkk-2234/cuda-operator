#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==================== 三种 kernel ====================

// 版本1：naive，每线程负责一个元素
__global__ void add_naive(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C, int M, int N) {
    int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) C[idx] = A[idx] + B[idx];
}

// 版本2：grid-stride 标量，固定线程数循环遍历
__global__ void add_grid_stride_scalar(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C, int M, int N) {
    int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// 版本3：grid-stride + float4 向量化，一次读写 4 个 float
__global__ void add_grid_stride_vec4(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C, int M, int N) {
    int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    int N4 = total / 4;
    for (int i = idx; i < N4; i += stride) {
        float4 a = __ldg(A4 + i);
        float4 b = __ldg(B4 + i);
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[i] = c;
    }

    // 尾部不足 4 个的元素，标量兜底
    int remain = N4 * 4;
    for (int i = remain + idx; i < total; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// ==================== 启动封装 ====================

void launch_naive(const float* a, const float* b, float* c, int M, int N) {
    int block = 256;
    int total = M * N;
    int grid = (total + block - 1) / block;
    add_naive<<<grid, block>>>(a, b, c, M, N);
}

void launch_grid_stride_scalar(const float* a, const float* b, float* c, int M, int N) {
    int block = 256;
    int total = M * N;
    int grid = min((total + block - 1) / block, 1024);
    add_grid_stride_scalar<<<grid, block>>>(a, b, c, M, N);
}

void launch_grid_stride_vec4(const float* a, const float* b, float* c, int M, int N) {
    int block = 256;
    int total = M * N;
    int grid = min((total + block - 1) / block, 1024);
    add_grid_stride_vec4<<<grid, block>>>(a, b, c, M, N);
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
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    add_naive<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, N);
    cudaDeviceSynchronize();
}

// ==================== main ====================

int main() {
    // ==================== 正确性校验 ====================
    int pass = 1;
    pass &= verify(launch_naive, "add_naive");
    pass &= verify(launch_grid_stride_scalar, "add_grid_stride_scalar");
    pass &= verify(launch_grid_stride_vec4, "add_grid_stride_vec4");
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
    benchmark(launch_naive, d_a, d_b, d_c, M, N, "add_naive");
    benchmark(launch_grid_stride_scalar, d_a, d_b, d_c, M, N, "add_grid_stride_scalar");
    benchmark(launch_grid_stride_vec4, d_a, d_b, d_c, M, N, "add_grid_stride_vec4");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    return 0;
}