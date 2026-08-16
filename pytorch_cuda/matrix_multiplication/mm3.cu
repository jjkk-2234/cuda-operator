#include <cuda_runtime.h>
// BM是对M分块的大小，BN是对N分块的大小，BK是对K分块的大小，TM是一个线程处理的M大小（即一个线程处理多少行数据）
template<int BM = 128, int BN = 64, int BK = 4, int TM = 16>
__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    constexpr int NT = (BM / TM) * BN; // NT是线程的数量
    __shared__ float SA[BM][BK], SB[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    float sum[TM] = {0.0f};

    for (int tile = 0; tile < K; tile += BK) {
        #pragma unroll
        // 从A中读取BM行BK列的数据到SA中, i+=NT表示搬运下一批次数据（指的是如果tile数据大小大于NT的话，每个线程要多搬几次）
        for (int i = tid; i < BM * BK; i += NT) {
            int load_a_row = i / BK;
            int load_a_col = i % BK;
            int ga_row = blockIdx.y * BM + load_a_row;
            int ga_col = tile + load_a_col;
            SA[load_a_row][load_a_col] = (ga_row < M && ga_col < K) ? A[ga_row * K + ga_col] : 0.0f;
        }

        #pragma unroll
        for (int i = tid; i < BK * BN; i += NT) {
            int load_b_row = i / BN;
            int load_b_col = i % BN;
            int gb_row = tile + load_b_row;
            int gb_col = blockIdx.x * BN + load_b_col;
            SB[load_b_row][load_b_col] = (gb_row < K && gb_col < N) ? B[gb_row * N + gb_col] : 0.0f;
        }
        __syncthreads();
        // 关键步骤：每个线程处理TM行数据
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float b_val = SB[k][tx];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                sum[m] += SA[ty * TM + m][k] * b_val;
            }
        }
        __syncthreads();
    }

    int col = blockIdx.x * BN + tx;
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int row = blockIdx.y * BM + ty * TM + m;
        if (row < M && col < N) {
            C[row * N + col] = sum[m];
        }
    } 
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BN, BM / TM);
    dim3 blocksPerGrid((N + BN - 1) / BN,
                       (M + BM - 1) / BM);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}