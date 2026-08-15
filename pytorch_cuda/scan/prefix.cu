#include <cuda_runtime.h>
__global__ void scan_v0_naive(const float* in, float* out, int n) {
    extern __shared__ float buf[];

    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    buf[pout * n + tid] = (tid > 0) ? in[tid - 1] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        pout = 1 - pout; // swap double buffer indices
        pin = 1 - pout;
        buf[pout * n + tid] = buf[pin * n + tid];

        if (tid >= offset)
            buf[pout * n + tid] += buf[pin * n + tid - offset];
        __syncthreads();
    }
    out[tid] = buf[pout * n + tid];
}

inline void launch_v0(const float* in, float* out, int n){
    scan_v0_naive<<<1, n, 2 * n * sizeof(float)>>>(in, out, n);
}