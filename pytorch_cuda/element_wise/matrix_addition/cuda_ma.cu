#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
// 直接加法核函数
__global__ void add_naive(const float* __restrict__ A, 
                          const float* __restrict__ B, 
                          float* __restrict__ C, int N) {
    int total = N * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void add_grid_stride_scalar(const float* __restrict__ A, 
                                       const float* __restrict__ B, 
                                       float* __restrict__ C, int N) {
    int total = N * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = idx; idx < total; idx += stride) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void add_grid_stride_vec4(const float* __restrict__ A, 
                                     const float* __restrict__ B, 
                                     float* __restrict__ C, int N) {
    int total = N * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);
    int N4 = total / 4;
    for (int i = idx; i < N4; i += strider) {
        float4 a = __ldg(A4 + i);
        float4 b = __ldg(B4 + i);
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[i] = c;
    }
    int remain = N4 * 4;
    for (int i = remain + idx; i < total; i += stride) {
        C[i] = A[i] + B[i];
    }
}

__global__ void add_2d_grid_stride(const float* __restrict__ A, 
                                 const float* __restrict__ B, 
                                 float* __restrict__ C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_y = blockDim.y * gridDim.y;
    int stride_x = blockDim.x * gridDim.x;
    for (int r = row; r < N; r += stride_y) {
        for (int c = col; c < N; c += stride_x) {
            int idx = r * N + c;
            C[idx] = A[idx] + B[idx];
        }
    }
}