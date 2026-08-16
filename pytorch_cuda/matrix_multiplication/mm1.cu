#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    // A: M x K, B: K x N, C: M x N
    // 假设M=N=K=4096=2^12=4KB
    // 一个线程处理一个计算结果, 则一个线程读取了A的行和B的列
    // 每行处理了K个元素, 每列处理了K个元素, 所以每个线程读取了2*K=8192=4KB个元素
    // 一个线程块要处理8192*1024=8388608=2^23=8MB个元素
    // 根据计算ceil(4096/32)=128, 所以需要128*128=2^14=16384个线程块
    // 整个kernel处理了128*128*8388608=2^37=128GB=的内存
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
