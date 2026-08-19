#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==================== 三种 kernel ====================

// 版本1：naive，每线程负责一个元素
__global__ void reverse_naive(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 2) {
        float tmp = input[idx];
        input[idx] = input[N - 1 - idx];
        input[N - 1 - idx] = tmp;
    }
}

// 版本2：grid-stride 标量，固定线程数循环遍历
__global__ void reverse_grid_stride_scalar(float* input, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n = N >> 1;
    for (int i = idx; i < n; i += stride) {
        float tmp = input[i];
        input[i] = input[N - 1 - i];
        input[N - 1 - i] = tmp;
    }
}

// 版本3：grid-stride + float4 向量化交换
__global__ void reverse_grid_stride_vec4(float* input, int N) {
    float4* in4 = reinterpret_cast<float4*>(input);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int n = N >> 1;
    int n4 = n >> 2;
    for (int i = idx; i < n4; i += stride) {
        float4 vec4 = in4[i];
        float temp = vec4.x;
        int id = i * 4;
        vec4.x = input[N - 1 - id];
        input[N - 1 - id] = temp;

        temp = vec4.y;
        id++;
        vec4.y = input[N - 1 - id];
        input[N - 1 - id] = temp;

        temp = vec4.z;
        id++;
        vec4.z = input[N - 1 - id];
        input[N - 1 - id] = temp;

        temp = vec4.w;
        id++;
        vec4.w = input[N - 1 - id];
        input[N - 1 - id] = temp;

        in4[i] = vec4;
    }
    int remain = n4 << 2;
    for (int i = remain + idx; i < n; i += stride) {
        float temp = input[i];
        input[i] = input[N - 1 - i];
        input[N - 1 - i] = temp;
    }
}

// ==================== 启动封装 ====================

void launch_naive(float* input, int N) {
    int block = 256;
    int grid = (N / 2 + block - 1) / block;
    reverse_naive<<<grid, block>>>(input, N);
}

void launch_grid_stride_scalar(float* input, int N) {
    int block = 256;
    int grid = std::min((N / 2 + block - 1) / block, 1024);
    reverse_grid_stride_scalar<<<grid, block>>>(input, N);
}

void launch_grid_stride_vec4(float* input, int N) {
    int block = 256;
    int grid = std::min((N / 2 + block - 1) / block, 1024);
    reverse_grid_stride_vec4<<<grid, block>>>(input, N);
}

// ==================== 正确性校验 ====================

int verify(void (*launch)(float*, int), const char* name) {
    int Ns[] = {1, 3, 7, 8, 100, 1000, 4096, 8192};
    int n_cases = sizeof(Ns) / sizeof(Ns[0]);
    int ok = 1;

    for (int t = 0; t < n_cases; t++) {
        int N = Ns[t];
        int n = N >> 1;
        size_t bytes = (size_t)N * sizeof(float);

        float* h_input = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);
        for (int i = 0; i < N; i++) {
            h_input[i] = (float)rand() / RAND_MAX;
            h_ref[i] = h_input[i];
        }
        // 参考：原地反转
        for (int i = 0; i < n; i++) {
            float tmp = h_ref[i];
            h_ref[i] = h_ref[N - 1 - i];
            h_ref[N - 1 - i] = tmp;
        }

        float* d_input;
        cudaMalloc(&d_input, bytes);
        cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

        launch(d_input, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_input, d_input, bytes, cudaMemcpyDeviceToHost);

        double max_diff = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = fabs((double)h_input[i] - (double)h_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff > 1e-6) {
            printf("%-24s  N=%5d  FAIL  max_diff=%.3e\n", name, N, max_diff);
            ok = 0;
        }

        cudaFree(d_input);
        free(h_input);
        free(h_ref);
    }
    if (ok) printf("%-24s  ALL PASS (%d 种形状)\n", name, n_cases);
    return ok;
}

// ==================== 计时 + 带宽 ====================

void benchmark(void (*launch)(float*, int), float* input, int N, const char* name) {
    int n_warmup = 5, n_repeat = 20;
    for (int i = 0; i < n_warmup; i++) launch(input, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_repeat; i++) launch(input, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= n_repeat;

    // 反转：读 N + 写 N = 2 * N * 4 字节
    float gb = 2.0f * N * sizeof(float) / 1e9f;
    printf("%-24s  %8.3f ms   %8.1f GB/s\n", name, ms, gb / (ms / 1000.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ==================== main ====================

int main() {
    // ==================== 正确性校验 ====================
    int pass = 1;
    pass &= verify(launch_naive, "reverse_naive");
    pass &= verify(launch_grid_stride_scalar, "reverse_grid_stride_scalar");
    pass &= verify(launch_grid_stride_vec4, "reverse_grid_stride_vec4");
    printf("\n%s\n\n", pass ? "正确性校验：ALL PASS" : "正确性校验：SOME FAILED");
    if (!pass) return 1;

    // ==================== 性能测试 ====================
    int N = 1 << 26;  // 2^26 个元素，约 268MB
    size_t bytes = (size_t)N * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    float* d_input;
    cudaMalloc(&d_input, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    printf("N = %d (%.2f GB)\n\n", N, bytes / 1e9f);
    benchmark(launch_naive, d_input, N, "reverse_naive");
    benchmark(launch_grid_stride_scalar, d_input, N, "reverse_grid_stride_scalar");
    benchmark(launch_grid_stride_vec4, d_input, N, "reverse_grid_stride_vec4");

    cudaFree(d_input);
    free(h_input);
    return 0;
}
