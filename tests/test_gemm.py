import pytest

from tensorforge.gemm import DType, Gemm


def test_known_value_m2_n3_k4_fp32():
    # Hand calculation:
    #   A elements = M*K = 8,  A bytes = 8*4  = 32
    #   B elements = K*N = 12, B bytes = 12*4 = 48
    #   C elements = M*N = 6,  C bytes = 6*4  = 24
    #   DRAM bytes = 32 + 48 + 24 = 104
    #   MACs = M*N*K = 24
    #   FLOPs = 2*MACs = 48
    gemm = Gemm(m=2, n=3, k=4, dtype=DType.FP32)

    assert gemm.a_elements == 8
    assert gemm.b_elements == 12
    assert gemm.c_elements == 6

    assert gemm.a_bytes == 32
    assert gemm.b_bytes == 48
    assert gemm.c_bytes == 24
    assert gemm.dram_bytes == 104

    assert gemm.macs == 24
    assert gemm.flops == 48


def test_dtype_changes_bytes_not_operation_count():
    fp32 = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
    fp16 = Gemm(m=2, n=3, k=4, dtype=DType.FP16)
    int8 = Gemm(m=2, n=3, k=4, dtype=DType.INT8)

    assert fp32.macs == fp16.macs == int8.macs == 24
    assert fp32.flops == fp16.flops == int8.flops == 48

    assert fp32.dram_bytes == 104
    assert fp16.dram_bytes == 52
    assert int8.dram_bytes == 26


def test_scaling_doubling_all_dims_gives_8x_macs():
    small = Gemm(m=64, n=64, k=64, dtype=DType.FP32)
    large = Gemm(m=128, n=128, k=128, dtype=DType.FP32)

    assert large.macs == small.macs * 8
    assert large.flops == small.flops * 8


@pytest.mark.parametrize("m,n,k", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1)])
def test_rejects_nonpositive_dimensions(m, n, k):
    with pytest.raises(ValueError):
        Gemm(m=m, n=n, k=k, dtype=DType.FP32)
