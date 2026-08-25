"""Analytical model of a single GEMM operation: C[M,N] = A[M,K] x B[K,N].

Operation convention (documented, not implicit):
    MACs  = M * N * K
    FLOPs = 2 * MACs        (1 multiply-accumulate = 2 floating-point operations)

Memory-traffic convention for Milestone 1 (idealized, not a cache/dataflow
simulation):
    A is read once from the modeled memory boundary
    B is read once from the modeled memory boundary
    C is written once to the modeled memory boundary

    DRAM bytes = bytes(A) + bytes(B) + bytes(C)

This assumes no tiling, no reuse beyond a single full-tensor pass, and no
SRAM capacity limits. Later milestones (tiling, dataflow) will replace this.
"""

from dataclasses import dataclass
from enum import Enum


class DType(Enum):
    """Element data type, represented only by its width in bytes."""

    FP32 = 4
    FP16 = 2
    INT8 = 1

    @property
    def bytes_per_element(self) -> int:
        return self.value


@dataclass(frozen=True)
class Gemm:
    """A single dense matrix multiply C = A x B.

    A: M x K
    B: K x N
    C: M x N
    """

    m: int
    n: int
    k: int
    dtype: DType

    def __post_init__(self) -> None:
        for name, value in (("m", self.m), ("n", self.n), ("k", self.k)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

    @property
    def macs(self) -> int:
        """Multiply-accumulate count: M * N * K."""
        return self.m * self.n * self.k

    @property
    def flops(self) -> int:
        """Floating-point operation count: 2 * MACs (1 MAC = 2 FLOPs)."""
        return 2 * self.macs

    @property
    def a_elements(self) -> int:
        return self.m * self.k

    @property
    def b_elements(self) -> int:
        return self.k * self.n

    @property
    def c_elements(self) -> int:
        return self.m * self.n

    @property
    def a_bytes(self) -> int:
        return self.a_elements * self.dtype.bytes_per_element

    @property
    def b_bytes(self) -> int:
        return self.b_elements * self.dtype.bytes_per_element

    @property
    def c_bytes(self) -> int:
        return self.c_elements * self.dtype.bytes_per_element

    @property
    def dram_bytes(self) -> int:
        """Idealized modeled DRAM traffic: A read once + B read once + C written once.

        This is an analytical simplifying assumption, not a measurement of
        actual reuse, tiling, or cache behavior.
        """
        return self.a_bytes + self.b_bytes + self.c_bytes
