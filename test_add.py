
import torch
import pytest

def torch_add(x0, x1):
    res = x0 + x1
    return res


#####################################################################
######################## test case 1 ################################
#####################################################################
import triton
import triton.language as tl


@triton.jit
def triton_add_1(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    tl.device_print("pid: ", pid)
    offset = pid * XBLOCK
    base = tl.arange(0, XBLOCK)
    x0 = offset + base
    tmp0 = tl.load(in_ptr0 + x0)
    tmp1 = tl.load(in_ptr1 + x0)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2)


# pytest -sv 01_add.py::test_case1
def test_case1():
    dtype = torch.float32
    device = 'npu'

    x0 = torch.randn(4, 10, 512, dtype=dtype, device=device)
    x1 = torch.randn(4, 10, 512, dtype=dtype, device=device)

    y_ref = torch_add(x0, x1)
    y_cal = torch.zeros(x0.shape, dtype=dtype, device=device)

    # 核间切分
    # ncore = 1
    # xblock = 4*10*512

    # ncore = 4
    # xblock = 10*512

    ncore = 40
    xblock = 512

    triton_add_1[ncore, 1, 1](x0, x1, y_cal, xblock)


    torch.testing.assert_close(y_cal, y_ref, atol=1e-5, rtol=1e-5)
    print("Test Add Case1 PASSED")



#####################################################################
######################## test case 2 ################################
#####################################################################
@triton.jit
def triton_add_2(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops = XBLOCK // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        tmp0 = tl.load(in_ptr0 + x0)
        tmp1 = tl.load(in_ptr1 + x0)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + x0, tmp2)


# pytest -sv 01_add.py::test_case2
def test_case2():
    dtype = torch.float32
    device = 'npu'

    x0 = torch.randn(4, 10, 1024*48, dtype=dtype, device=device)
    x1 = torch.randn(4, 10, 1024*48, dtype=dtype, device=device)

    y_ref = torch_add(x0, x1)
    y_cal = torch.zeros(x0.shape, dtype=dtype, device=device)

    # 核间切分
    ncore = 40
    xblock = 1024*48
    # triton_add_1[ncore, 1, 1](x0, x1, y_cal, xblock)


    # 核内切分
    xblock_sub = 1024
    triton_add_2[ncore, 1, 1](x0, x1, y_cal, xblock, xblock_sub)

    torch.testing.assert_close(y_cal, y_ref, atol=1e-5, rtol=1e-5)
    print("Test Add Case2 PASSED")



#####################################################################
######################## test case 3 ################################
#####################################################################
@triton.jit
def triton_add_3(in_ptr0, in_ptr1, out_ptr0, numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    loops = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop in range(loops):
        x0 = offset + (loop * XBLOCK_SUB) + base
        mask = x0 < numel
        tmp0 = tl.load(in_ptr0 + x0, mask=mask, other=0.0)
        tmp1 = tl.load(in_ptr1 + x0, mask=mask, other=0.0)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + x0, tmp2, mask=mask)


# pytest -sv 01_add.py::test_case3
def test_case3():
    dtype = torch.float32
    device = 'npu'

    x0 = torch.randn(4, 1024*48+1, dtype=dtype, device=device)
    x1 = torch.randn(4, 1024*48+1, dtype=dtype, device=device)

    y_ref = torch_add(x0, x1)
    y_cal = torch.zeros(4, 1024*48+1, dtype=dtype, device=device)

    ncore = 4
    xblock = 1024*48+1 # 核间切分
    xblock_sub = 1024 # 核内切分

    # triton_add_2[ncore, 1, 1](x0, x1, y_cal, xblock, xblock_sub)

    numel = 4*(1024*48+1) # 用于计算mask
    triton_add_3[ncore, 1, 1](x0, x1, y_cal, numel, xblock, xblock_sub)

    torch.testing.assert_close(y_cal, y_ref, atol=1e-5, rtol=1e-5)
    print("Test Add Case3 PASSED")