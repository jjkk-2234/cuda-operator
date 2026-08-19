#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ==================== Warp / Block 归约 ====================

__inline__ __device__ float warpReduceMax(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warpReduceMax(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = blockDim.x >> 5;
    val = (threadIdx.x < numWarps) ? smem[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    int numWarps = blockDim.x >> 5;
    val = (threadIdx.x < numWarps) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ==================== 2D 版本1：online 单趟 ====================

__global__ void softmax_2d_online(const float* x, float* y, int n_cols) {
    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = &shared[1];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    const float* row_x = x + row * n_cols;
    float* row_y = y + row * n_cols;
    float mm = -FLT_MAX, ss = 0.0f;
    for (int i = tid; i < n_cols; i += stride) {
        float val = row_x[i];
        float mm_new = fmaxf(mm, val);
        if (mm > -FLT_MAX) ss *= __expf(mm - mm_new);
        ss += __expf(val - mm_new);
        mm = mm_new;
    }
    float block_max = blockReduceMax(mm);
    if (tid == 0) s_max[0] = block_max;
    __syncthreads();
    block_max = s_max[0];
    float block_sum = blockReduceSum(ss * __expf(mm - block_max));
    if (tid == 0) s_sum[0] = block_sum;
    __syncthreads();
    block_sum = s_sum[0];
    for (int i = tid; i < n_cols; i += stride) {
        row_y[i] = __expf(row_x[i] - block_max) / block_sum;
    }
}

// ==================== 2D 版本2：两阶段 ====================

__global__ void softmax_2d_two_pass(const float* x, float* y, int n_cols) {
    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = &shared[1];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    const float* row_x = x + row * n_cols;
    float* row_y = y + row * n_cols;
    float local_max = -FLT_MAX;
    for (int i = tid; i < n_cols; i += stride) {
        local_max = fmaxf(local_max, row_x[i]);
    }
    float block_max = blockReduceMax(local_max);
    if (tid == 0) s_max[0] = block_max;
    __syncthreads();
    block_max = s_max[0];
    float local_sum = 0.0f;
    for (int i = tid; i < n_cols; i += stride) {
        local_sum += __expf(row_x[i] - block_max);
    }
    float block_sum = blockReduceSum(local_sum);
    if (tid == 0) s_sum[0] = block_sum;
    __syncthreads();
    block_sum = s_sum[0];
    for (int i = tid; i < n_cols; i += stride) {
        row_y[i] = __expf(row_x[i] - block_max) / block_sum;
    }
}

// ==================== 2D 版本3：online + float4 ====================

__global__ void softmax_2d_vec4(const float* x, float* y, int n_cols) {
    extern __shared__ float shared[];
    float* s_max = shared;
    float* s_sum = &shared[1];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    const float* row_x = x + row * n_cols;
    float* row_y = y + row * n_cols;
    float mm = -FLT_MAX, ss = 0.0f;
    int n4 = n_cols >> 2;
    const float4* row_x_vec4 = reinterpret_cast<const float4*>(row_x);
    for (int i = tid; i < n4; i += stride) {
        float4 v = row_x_vec4[i];
        float mnew = fmaxf(mm, v.x);
        ss = ss * __expf(mm - mnew) + __expf(v.x - mnew); mm = mnew;
        mnew = fmaxf(mm, v.y);
        ss = ss * __expf(mm - mnew) + __expf(v.y - mnew); mm = mnew;
        mnew = fmaxf(mm, v.z);
        ss = ss * __expf(mm - mnew) + __expf(v.z - mnew); mm = mnew;
        mnew = fmaxf(mm, v.w);
        ss = ss * __expf(mm - mnew) + __expf(v.w - mnew); mm = mnew;
    }
    for (int i = (n4 << 2) + tid; i < n_cols; i += stride) {
        float mnew = fmaxf(mm, row_x[i]);
        ss = ss * __expf(mm - mnew) + __expf(row_x[i] - mnew);
        mm = mnew;
    }
    float block_max = blockReduceMax(mm);
    if (tid == 0) s_max[0] = block_max;
    __syncthreads();
    block_max = s_max[0];
    float block_sum = blockReduceSum(ss * __expf(mm - block_max));
    if (tid == 0) s_sum[0] = block_sum;
    __syncthreads();
    block_sum = s_sum[0];
    float4* row_y_vec4 = reinterpret_cast<float4*>(row_y);
    for (int i = tid; i < n4; i += stride) {
        float4 v = row_x_vec4[i];
        v.x = __expf(v.x - block_max) / block_sum;
        v.y = __expf(v.y - block_max) / block_sum;
        v.z = __expf(v.z - block_max) / block_sum;
        v.w = __expf(v.w - block_max) / block_sum;
        row_y_vec4[i] = v;
    }
    for (int i = (n4 << 2) + tid; i < n_cols; i += stride) {
        row_y[i] = __expf(row_x[i] - block_max) / block_sum;
    }
}

// ==================== 启动封装 ====================

void launch_2d_online(const float* x, float* y, int n_rows, int n_cols) {
    int block = 256;
    int shared = 2 * sizeof(float);
    softmax_2d_online<<<n_rows, block, shared>>>(x, y, n_cols);
}

void launch_2d_two_pass(const float* x, float* y, int n_rows, int n_cols) {
    int block = 256;
    int shared = 2 * sizeof(float);
    softmax_2d_two_pass<<<n_rows, block, shared>>>(x, y, n_cols);
}

void launch_2d_vec4(const float* x, float* y, int n_rows, int n_cols) {
    int block = 256;
    int shared = 2 * sizeof(float);
    softmax_2d_vec4<<<n_rows, block, shared>>>(x, y, n_cols);
}

// ==================== 正确性校验 ====================

int verify(void (*launch)(const float*, float*, int, int), const char* name) {
    int rows[] = {1, 100, 1000};
    int cols[] = {100, 1000, 4096};
    int n_cases = sizeof(rows) / sizeof(rows[0]);
    int ok = 1;
    for (int t = 0; t < n_cases; t++) {
        int M = rows[t], N = cols[t];
        int total = M * N;
        size_t bytes = (size_t)total * sizeof(float);
        float* h_x = (float*)malloc(bytes);
        float* h_ref = (float*)malloc(bytes);
        float* h_y = (float*)malloc(bytes);
        for (int i = 0; i < total; i++) h_x[i] = (float)rand() / RAND_MAX;
        for (int r = 0; r < M; r++) {
            float max_val = -FLT_MAX, sum_exp = 0.0f;
            for (int c = 0; c < N; c++) max_val = fmaxf(max_val, h_x[r * N + c]);
            for (int c = 0; c < N; c++) sum_exp += expf(h_x[r * N + c] - max_val);
            for (int c = 0; c < N; c++) h_ref[r * N + c] = expf(h_x[r * N + c] - max_val) / sum_exp;
        }
        float* d_x, *d_y;
        cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
        cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
        launch(d_x, d_y, M, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
        double max_diff = 0.0;
        for (int i = 0; i < total; i++) {
            double diff = fabs((double)h_y[i] - (double)h_ref[i]);
            if (diff > max_diff) max_diff = diff;
        }
        if (max_diff > 1e-5) {
            printf("%-24s  %dx%d  FAIL  max_diff=%.3e\n", name, M, N, max_diff);
            ok = 0;
        }
        cudaFree(d_x); cudaFree(d_y);
        free(h_x); free(h_ref); free(h_y);
    }
    if (ok) printf("%-24s  ALL PASS (%d 种形状)\n", name, n_cases);
    return ok;
}

// ==================== 计时 + 带宽 ====================

void benchmark(void (*launch)(const float*, float*, int, int), const float* x, float* y, int M, int N, const char* name) {
    int n_warmup = 5, n_repeat = 20;
    for (int i = 0; i < n_warmup; i++) launch(x, y, M, N);
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < n_repeat; i++) launch(x, y, M, N);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop); ms /= n_repeat;
    float gb = 2.0f * M * N * sizeof(float) / 1e9f;
    printf("%-24s  %8.3f ms   %8.1f GB/s\n", name, ms, gb / (ms / 1000.0f));
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// ==================== main ====================

int main() {
    int pass = 1;
    pass &= verify(launch_2d_online, "softmax_2d_online");
    pass &= verify(launch_2d_two_pass, "softmax_2d_two_pass");
    pass &= verify(launch_2d_vec4, "softmax_2d_vec4");
    printf("\n%s\n\n", pass ? "正确性校验：ALL PASS" : "正确性校验：SOME FAILED");
    if (!pass) return 1;
    int M = 8192, N = 8192;
    int total = M * N;
    size_t bytes = (size_t)total * sizeof(float);
    float* h_x = (float*)malloc(bytes);
    for (int i = 0; i < total; i++) h_x[i] = (float)rand() / RAND_MAX;
    float* d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    printf("M = %d, N = %d (%.2f GB)\n\n", M, N, bytes / 1e9f);
    benchmark(launch_2d_online, d_x, d_y, M, N, "softmax_2d_online");
    benchmark(launch_2d_two_pass, d_x, d_y, M, N, "softmax_2d_two_pass");
    benchmark(launch_2d_vec4, d_x, d_y, M, N, "softmax_2d_vec4");
    cudaFree(d_x); cudaFree(d_y); free(h_x);
    return 0;
}
