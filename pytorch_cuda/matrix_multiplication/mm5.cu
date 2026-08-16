#include <cuda_runtime.h>
// A: M x N, B: N x K, C: M x K
// BM是对M分块的大小，BN是对输出列（K）分块的大小，BK是对内积维（N）分块的大小，TM是一个线程处理的M大小（即一个线程处理多少行数据）
template<int BM = 128, int BN = 128, int BK = 16, int TM = 8, int TN = 8>
__global__ void matrix_multiplication_kernel(
    const __restrict__ float* A, 
    const __restrict__ float* B, 
    __restrict__ float* C, 
    int M, int N, int K) {
    constexpr int NT = (BM / TM) * (BN / TN); // NT是线程的数量
    __shared__ float SA[BM][BK], SB[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    float sum[TM][TN] = {{0.0f}};

    for (int tile = 0; tile < (N + BK - 1) / BK; tile++) {
        int gk = tile * BK;

        #pragma unroll
        for (int i = tid; i < BM * BK; i += NT) {
            int load_a_row = i / BK;
            int load_a_col = i % BK;
            int ga_row = blockIdx.y * BM + load_a_row;
            int ga_col = gk + load_a_col;
            SA[load_a_row][load_a_col] = (ga_row < M && ga_col < N) ? A[ga_row * N + ga_col] : 0.0f;
        }

        #pragma unroll
        for (int i = tid; i < BK * BN; i += NT) {
            int load_b_row = i / BN;
            int load_b_col = i % BN;
            int gb_row = gk + load_b_row;
            int gb_col = blockIdx.x * BN + load_b_col;
            SB[load_b_row][load_b_col] = (gb_row < N && gb_col < K) ? B[gb_row * K + gb_col] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int m = 0; m < TM; m++) a_reg[m] = SA[ty * TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) b_reg[n] = SB[k][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    sum[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = blockIdx.y * BM + ty * TM + m;
        if (row >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int col = blockIdx.x * BN + tx * TN + n;
            if (col < K) {
                C[row * K + col] = sum[m][n];
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BN / TN, BM / TM);
    dim3 blocksPerGrid((K + BN - 1) / BN,
                       (M + BM - 1) / BM);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}