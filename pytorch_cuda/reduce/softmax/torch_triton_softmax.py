# import triton
# import triton.language as tl
import torch
# torch version
def softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True)[0]

    safe_x = x - x_max

    exp_x = torch.exp(safe_x)

    sum_exp = exp_x.sum(dim=dim, keepdim=True)

    output = exp_x / sum_exp

    return output

# triton version
# 这个计算方法类似cuda中的warp shuffle，在寄存器内完成运算
# 只进行了一次全局内存的读和写，减少了内存访问次数，提高了性能
@triton.jit
def kernel_softmax_fuse(
    x_ptr, x_row_stride,
    y_ptr, y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)

    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    idx = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float('inf'))
    x = tl.exp(x - tl.max(x, axis=0))
    eps = float(1e-9)
    x /= tl.maximum(tl.sum(x), eps)
    tl.store(y_ptr + idx, x, mask = idx < n_cols)

def triton_softmax_dim1_fuse(x):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_fuse[[n_rows]](x, x.stride(0), y, y.stride(0), n_cols, BLOCK_SIZE=triton.next_power_of_2(n_cols), num_warps=32)
    return y

@triton.jit
def kernel_softmax_tile(
    x_ptr, x_row_stride,
    y_ptr, y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride
    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")

    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        # x.shape = (BLOCK_SIZE, 1)
        x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float('inf'))
        # 先用向量的方式求最大值，怎么求的？
        # 两个向量的每个分量都取最大值，然后再拼成一个新的向量
        # mm.shape = (BLOCK_SIZE, 1)
        mm = tl.maximum(mm, x)
    # 最后用reduce的方式求最大值
    # mm.shape = (1,)
    mm = tl.max(mm)
       
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # 这段分支语句的代码是什么意思？
    # 缓存预热	倒序处理时，前面的块（更可能被再次访问）最后处理，更可能留在缓存中
    # 写回优化	第三遍写回是正序的，倒序处理的块在写回时可能还在缓存中
    # 减少缓存miss	倒序可以减少后续访问时的缓存miss
    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            # x.shape = (BLOCK_SIZE, 1)
            x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
            # x.shape = (BLOCK_SIZE, 1)
            x = tl.exp(x - mm)
            # ss.shape = (BLOCK_SIZE, 1)
            ss += x
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm)
            ss += x
    ss = tl.sum(ss)
    eps = float(1e-9)
    ss = tl.maximum(ss, eps)

    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
        x = tl.exp(x - mm) / ss
        tl.store(y_ptr + idx, x, mask = idx < n_cols)

def triton_softmax_dim1_tile(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_tile[[n_rows]](x, x.stride(0), y, y.stride(0), n_cols, BLOCK_SIZE=2**14, CACHE_OPT=cache_opt, num_warps=32)
    return y

@triton.jit
def kernel_softmax_online(
    x_ptr, x_row_stride,
    y_ptr, y_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    CACHE_OPT: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_ptr += row_idx * x_row_stride
    y_ptr += row_idx * y_row_stride

    mm = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    ss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
        idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
        # mm_new.shape = (BLOCK_SIZE, 1), 每个分量包含一个最大值
        mm_new = tl.maximum(mm, x)
        # 第0轮不用修正
        if i:
            # ss.shape = (BLOCK_SIZE, 1)，在各自维度上修正ss
            ss *= tl.exp(mm - mm_new)
        # x.shape = (BLOCK_SIZE, 1), 在各自维度上去求指数值
        x = tl.exp(x - mm_new)
        # ss.shape = (BLOCK_SIZE, 1)，在各自维度上累加新的x指数值
        ss += tl.where(idx < n_cols, x, 0.0)
        mm = mm_new
    # 最后用reduce方式把各个维度合并起来
    mm_new = tl.max(mm)
    ss *= tl.exp(mm - mm_new)
    ss = tl.sum(ss)
    mm = mm_new

    eps = float(1e-9)
    ss = tl.maximum(ss, eps)
    if CACHE_OPT:
        for i in range(tl.cdiv(n_cols, BLOCK_SIZE) - 1, -1, -1):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask = idx < n_cols)
    else:
        for i in range(0, tl.cdiv(n_cols, BLOCK_SIZE)):
            idx = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
            x = tl.load(x_ptr + idx, mask = idx < n_cols, other=-float("inf"))
            x = tl.exp(x - mm) / ss
            tl.store(y_ptr + idx, x, mask = idx < n_cols)



def triton_softmax_dim1_online(x, cache_opt=True):
    n_rows, n_cols = x.shape
    y = torch.empty_like(x)
    kernel_softmax_online[[n_rows]](
        x, x.stride(0),
        y, y.stride(0),
        n_cols,
        BLOCK_SIZE=2**12,
        CACHE_OPT=cache_opt,
        num_warps=32,
    )
    return y

# if __name__ == '__main__':
#     x1 = torch.tensor([1.0, 2.0, 3.0])
#     p1 = softmax(x1)
#     print(torch.softmax(x1, dim=-1))
