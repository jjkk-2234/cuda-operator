#include <cuda_runtime.h>
#include <iostream>

constexpr int NUM_BANKS = 32;
constexpr int LOG_NUM_BANKS = 5;
// pad函数的作用：为了避免bank冲突，对索引进行偏移
// 举个例子？比如，当i=32时，i >> LOG_NUM_BANKS = 1，所以pad(32) = 33, 这样就避免了bank冲突
// 再举个例子？比如，当i=64时，i >> LOG_NUM_BANKS = 2，所以pad(64) = 66, 这样就避免了bank冲突
// 请问和transpose的什么地方相似？共享内存的数组大小为[32][33]，这样就避免了bank冲突
__device__ inline int pad(int i) {
    return i + (i >> LOG_NUM_BANKS);
}

__global__ void scan_v1_blelloch(const float* in, float* out, int n, int n2) {
    extern __shared__ float temp[];
    // 模拟特殊情况，n2=256, threads=128, 共享内存大小为33*sizeof(float)
    int thid = threadIdx.x;
    int offset = 1;
    // ai和bi分别表示当前线程处理的两个元素的索引
    int ai = thid;
    int bi = thid + (n2 >> 1);
    temp[pad(ai)] = (ai < n) ? in[ai] : 0.0f;
    temp[pad(bi)] = (bi < n) ? in[bi] : 0.0f;
    // upSweep阶段
    for (int d = n2 >> 1; d > 0; d >>= 1) {
        // d=128, offset=1
        // d=64, offset=2
        // d=32, offset=4
        // d=16, offset=8
        // d=8, offset=16
        // d=4, offset=32
        // d=2, offset=64
        // d=1, offset=128
        // 为什么这里要线程同步？
        // 因为在上一轮中，每个线程都更新了共享内存中的值，
        // 而这里需要等待所有线程都更新了共享内存中的值，才能继续执行
        __syncthreads();
        // 这个部分有warp divergence吗？有
        if (thid < d) {
            // 当offset提升到32时，d=4
            // 当thid=0时
            // left_idx = 32 * (2 * 0 + 1) - 1 = 31
            // right_idx = 32 * (2 * 0 + 2) - 1 = 63 = 32
            // 当thid=1时
            // left_idx = 32 * (2 * 1 + 1) - 1 = 95
            // right_idx = 32 * (2 * 1 + 2) - 1 = 127
            int left_idx = offset * (2 * thid + 1) - 1;
            int right_idx = offset * (2 * thid + 2) - 1;
            // 当thid=0时
            // pad_left = pad(31) = 31 + (31 >> 5) = 31 + 0 = 31
            // pad_right = pad(63) = 63 + (63 >> 5) = 63 + 1 = 64
            // 当thid=1时
            // pad_left = pad(95) = 95 + (95 >> 5) = 95 + 2 = 97
            // pad_right = pad(127) = 127 + (127 >> 5) = 127 + 3 = 130
            // 请问线程0和线程1发生bank conflict? 不会 31和97没有发生访问不同的bank
            // 如果没有pad函数，线程0和线程1会发生bank conflict吗？会 31和95发生访问同一个bank，即bank 31
            temp[pad(right_idx)] += temp[pad(left_idx)];
        }
        offset <<= 1;
    }
    // 初始化共享内存的最后一个元素为0
    if (thid == 0) {
        temp[pad(n2 - 1)] = 0.0f;
    }
    // downSweep阶段
    for (int d = 1; d < n2; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int left_idx = offset * (2 * thid + 1) - 1;
            int right_idx = offset * (2 * thid + 2) - 1;
            
            int pad_left = pad(left_idx);
            int pad_right = pad(right_idx);

            float t = temp[pad_left];
            temp[pad_left] = temp[pad_right];
            temp[pad_right] += t;
        }
    }
    __syncthreads();

    if (ai < n) out[ai] = temp[pad(ai)];
    if (bi < n) out[bi] = temp[pad(bi)];
}

inline void launch_v1(const float* in, float* out, int n) {
    // n2的作用：找到大于等于n的最小的2的幂次方，用于构建满二叉树
    int n2 = 1;
    while (n2 < n) n2 <<= 1;

    if (n2 > 2048) {
        std::cerr << "Error: n is too large for this kernel." << std::endl;
        return;
    }
    // 为什么threads=n2/2？
    // Blelloch 上扫/下扫的最大并行度出现在最宽层：上扫第一轮 d = n2/2，
    // 需要处理 n2/2 对元素，即最多同时有 n2/2 个线程有用。所以：
    int threads = (n2 / 2 > 0) ? n2 / 2 : 1;
    // 设置动态共享内存的大小
    // 比如n2=32，则padded_n=33，动态共享内存大小为33*sizeof(float)
    int padded_n = pad(n2);

    scan_v1_blelloch<<<1, threads, padded_n * sizeof(float)>>>(in, out, n, n2);
}