#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    uchar4* img4 = reinterpret_cast<uchar4*>(image);
    int num_pixels = width * height;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < num_pixels; i += stride) {
        uchar4 p = img4[i];
        p.x = 255 - p.x;
        p.y = 255 - p.y;
        p.z = 255 - p.z;
        img4[i] = p;
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = min((width * height + threadsPerBlock - 1) / threadsPerBlock, 1024);

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}