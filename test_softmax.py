import os

import torch
import torch_npu
import triton
import triton.language as tl
import triton.backends.ascend.runtime
from triton.backends.ascend.testing import do_bench_npu


def softmax_torch(x):
    return torch.softmax(x, axis=-1)

# 1.axis parse

# 2.gen tile

# 3.bench

@triton.autotune(
    configs=[],
    key=["n_rows", "n_cols"],
    # hints={
    #     "auto_gen_config": True,
    # }
)
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    # starting row of the program
    row_start = tl.program_id(0) * XBLOCK
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_offsets = row_start + row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]
        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols
        mask = xmask & ymask
        input_ptrs = input_ptr + (row_offsets * input_row_stride + col_offsets)
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=1).reshape(XBLOCK_SUB, 1).broadcast_to(
            XBLOCK_SUB, BLOCK_SIZE
        )
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = (
            tl.sum(numerator, axis=1)
            .reshape(XBLOCK_SUB, 1)
            .broadcast_to(XBLOCK_SUB, BLOCK_SIZE)
        )
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_ptrs = output_ptr + (row_offsets * output_row_stride + col_offsets)
        tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = n_cols

    # ncore = 1
    # XBLOCK = 10240
    # XBLOCK_SUB = 16

    ncore = 40
    XBLOCK = 256
    XBLOCK_SUB = 16

    # ncore = 80
    # XBLOCK = 128
    # XBLOCK_SUB = 16

    y = torch.empty_like(x)
    # grid = (ncore, 1, 1)

    grid = lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]), 1, 1)
    softmax_kernel[grid](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, 
        # XBLOCK=XBLOCK, XBLOCK_SUB=XBLOCK_SUB, 
        # multibuffer=False,
    )
    return y


# pytest -sv 02_softmax.py::test_softmax_acc
def test_softmax_acc():
    # shape = (16896, 1024)
    shape = (10240, 1024)
    x = torch.randn(shape, dtype=torch.float32, device="npu")

    y_torch = softmax_torch(x)
    y_triton = softmax_triton(x)
    assert torch.allclose(y_triton, y_torch)
    print(f"Fused Softmax Acc PASSED!")


# pytest -sv 02_softmax.py::test_softmax_perf
def test_softmax_perf():
    # shape = (16896, 1024)
    shape = (10240, 1024)
    x = torch.randn(shape, dtype=torch.float32, device="npu")
    time_eager = do_bench_npu(lambda: softmax_torch(x))
    time_triton = do_bench_npu(lambda: softmax_triton(x))

    # device_perf_npu(lambda: softmax_torch(x))
    # device_perf_npu(lambda: softmax_triton(x))

    print(f"{time_eager=} ms")
    print(f"{time_triton=} ms")
    assert (time_eager / time_triton) >= 1
    print(f"Fused Softmax Perf PASSED!")


def device_perf_npu(executor, profiling_dir="./npu_profiling", active=5):
    if not os.path.exists(profiling_dir):
        os.makedirs(profiling_dir)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        l2_cache=False,
        data_simplification=False,
    )

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=5, active=active, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        mat_a = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_b = torch.randn(4096, 4096).to(dtype=torch.bfloat16).npu()
        mat_c = torch.matmul(mat_a, mat_b)
        mat_c.cpu()

        for _ in range(10):
            executor()
            prof.step()
            torch.npu.synchronize()

# msprof op simulator --kernel-name=softmax_kernel --soc-version={soc-version} python 02_softmax.py
if __name__ == "__main__":
    # shape = (16896, 1024)
    shape = (10240, 1024)
    x = torch.randn(shape, dtype=torch.float32, device="npu")
    y_triton = softmax_triton(x)
    # y_torch = softmax_torch(x)