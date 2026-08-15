import torch
import triton
import triton.language as tl

@triton.jit
def invert_kernel(
    image_ptr,
    width,
    height,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    rx = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    ry = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)

    mask = (rx[:, None] < height) & (ry[None, :] < width)

    base_offsets = (rx[:, None] * width + ry[None, :]) * 4

    for i in range(3):
        ptr = image_ptr + base_offsets + i
        val = tl.load(ptr, mask=mask, other=0.0)
        tl.store(ptr, 255 - val, mask=mask)

def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE_X = 32
    BLOCK_SIZE_Y = 32

    grid = (
        triton.cdiv(height, BLOCK_SIZE_X),
        triton.cdiv(width, BLOCK_SIZE_Y),
    )
    invert_kernel[grid](image, width, height, BLOCK_SIZE_X, BLOCK_SIZE_Y)

    return image
