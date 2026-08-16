#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    // A: M x K, B: K x N, C: M x N
    constexpr int BM = 32, BN = 32, BK = 32;
    __shared__ float SA[BM][BK], SB[BK][BN];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row =  blockIdx.y * BM + ty;
    int col =  blockIdx.x * BN + tx;

    float sum = 0.0f;
    // 把A和B的tile切分块搬运到共享内存中
    for (int tile = 0; tile < K; tile += BK) {
        int ga = tile + tx;
        int gb = tile + ty;
        SA[ty][tx] = (row < M && ga < K) ? A[row * K + ga] : 0.0f;
        SB[ty][tx] = (gb < K && col < N) ? B[gb * N + col] : 0.0f;
        __syncthreads();
        // 计算C的tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // 在这个地方减少了全局内存的访问次数，减少的倍数大致为BM*BN倍
            sum += SA[ty][k] * SB[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
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