#include<torch/extension.h>
#include<cuda_runtime.h>

__global__ void add_kernel_vec4(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int N) {
    const float4* a4 = reinterpret_cast<const float4*>(a);
    const float4* b4 = reinterpret_cast<const float4*>(b);
    float4* c4 = reinterpret_cast<float4*>(c);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int N4 = N / 4;

    for (int i = idx; i < N4; i += stride) {
        float4 a_val = a4[i];
        float4 b_val = b4[i];
        float4 c_val;
        c_val.x = a_val.x + b_val.x;
        c_val.y = a_val.y + b_val.y;
        c_val.z = a_val.z + b_val.z;
        c_val.w = a_val.w + b_val.w;
        c4[i] = c_val;
    }
    // 这部分处理尾部剩余的元素
    int remainder_start = N4 * 4;
     
    for (int i = remainder_start + idx; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }

torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be a CUDA tensor");
    int N = a.numel();
    auto c = torch::empty_like(a);

    int threads_per_block = 256;
    int blocks_per_grid = (N / 4 + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid == 0) {
        blocks_per_grid = 1;
    }
    add_kernel_vec4<<<blocks_per_grid, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
    return c;
}

// 自动微分包装
class AddFunction : public torch::autograd::Function<AddFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor a,
        torch::Tensor b
    ) {
        ctx->save_for_backward({a, b});
        return add_forward(a, b);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto grad = grad_outputs[0];
        return {grad, grad};
    }

torch::Tensor add_autograd(torch::Tensor a, torch::Tensor b) {
    return AddFunction::apply(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_autograd, "Vector addition with autograd");
    m.def("add_forward", &add_forward, "Raw forward (no autograd)");
}