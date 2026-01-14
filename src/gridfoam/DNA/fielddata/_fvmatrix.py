from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.DNA.enum import Axis
from gridfoam.DNA.fielddata._cell import CellField


class FVMatrix:
    """
    A matrix representing the finite volume method.
    """

    def __init__(
        self, C: int, N: int, H: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        self._C = C
        self._N = N
        self._H = H
        self._dtype = dtype
        self._device = device
        self._a_P = CellField(1, C, N, H, dtype, device)
        self._a_E = CellField(1, C, N, H, dtype, device)
        self._a_W = CellField(1, C, N, H, dtype, device)
        self._a_N = CellField(1, C, N, H, dtype, device)
        self._a_S = CellField(1, C, N, H, dtype, device)
        self._a_T = CellField(1, C, N, H, dtype, device)
        self._a_B = CellField(1, C, N, H, dtype, device)
        self._source = CellField(1, C, N, H, dtype, device)

    @property
    def C(self) -> int:
        return self._C

    @property
    def N(self) -> int:
        return self._N

    @property
    def H(self) -> int:
        return self._H

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def a_P(self) -> CellField:
        return self._a_P

    @property
    def a_E(self) -> CellField:
        return self._a_E

    @property
    def a_W(self) -> CellField:
        return self._a_W

    @property
    def a_N(self) -> CellField:
        return self._a_N

    @property
    def a_S(self) -> CellField:
        return self._a_S

    @property
    def a_T(self) -> CellField:
        return self._a_T

    @property
    def a_B(self) -> CellField:
        return self._a_B

    @property
    def source(self) -> CellField:
        return self._source

    @a_P.setter
    def a_P(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        self._a_P.interior[0] = value

    @a_E.setter
    def a_E(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        self._a_E.interior[0] = value

    @a_W.setter
    def a_W(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        self._a_W.interior[0] = value

    @a_N.setter
    def a_N(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        self._a_N.interior[0] = value

    @a_S.setter
    def a_S(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        self._a_S.interior[0] = value

    @a_T.setter
    def a_T(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        self._a_T.interior[0] = value

    @a_B.setter
    def a_B(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        self._a_B.interior[0] = value

    @source.setter
    def source(self, value: Float[torch.Tensor, "C N N N"]) -> None:
        self._source.interior[0] = value

    def __iadd__(self, other: FVMatrix) -> FVMatrix:
        self._a_P.interior[0] += other._a_P.interior[0]
        self._a_E.interior[0] += other._a_E.interior[0]
        self._a_W.interior[0] += other._a_W.interior[0]
        self._a_N.interior[0] += other._a_N.interior[0]
        self._a_S.interior[0] += other._a_S.interior[0]
        self._a_T.interior[0] += other._a_T.interior[0]
        self._a_B.interior[0] += other._a_B.interior[0]
        self._source.interior[0] += other._source.interior[0]
        return self

    def __isub__(self, other: FVMatrix) -> FVMatrix:
        self._a_P.interior[0] -= other._a_P.interior[0]
        self._a_E.interior[0] -= other._a_E.interior[0]
        self._a_W.interior[0] -= other._a_W.interior[0]
        self._a_N.interior[0] -= other._a_N.interior[0]
        self._a_S.interior[0] -= other._a_S.interior[0]
        self._a_T.interior[0] -= other._a_T.interior[0]
        self._a_B.interior[0] -= other._a_B.interior[0]
        self._source.interior[0] -= other._source.interior[0]
        return self

    def __add__(self, other: FVMatrix) -> FVMatrix:
        result = FVMatrix(self.C, self.N, self.H, self.dtype, self.device)
        result._a_P.interior[0] = self._a_P.interior[0] + other._a_P.interior[0]
        result._a_E.interior[0] = self._a_E.interior[0] + other._a_E.interior[0]
        result._a_W.interior[0] = self._a_W.interior[0] + other._a_W.interior[0]
        result._a_N.interior[0] = self._a_N.interior[0] + other._a_N.interior[0]
        result._a_S.interior[0] = self._a_S.interior[0] + other._a_S.interior[0]
        result._a_T.interior[0] = self._a_T.interior[0] + other._a_T.interior[0]
        result._a_B.interior[0] = self._a_B.interior[0] + other._a_B.interior[0]
        result._source.interior[0] = (
            self._source.interior[0] + other._source.interior[0]
        )
        return result

    def __sub__(self, other: FVMatrix) -> FVMatrix:
        result = FVMatrix(self.C, self.N, self.H, self.dtype, self.device)
        result._a_P.interior[0] = self._a_P.interior[0] - other._a_P.interior[0]
        result._a_E.interior[0] = self._a_E.interior[0] - other._a_E.interior[0]
        result._a_W.interior[0] = self._a_W.interior[0] - other._a_W.interior[0]
        result._a_N.interior[0] = self._a_N.interior[0] - other._a_N.interior[0]
        result._a_S.interior[0] = self._a_S.interior[0] - other._a_S.interior[0]
        result._a_T.interior[0] = self._a_T.interior[0] - other._a_T.interior[0]
        result._a_B.interior[0] = self._a_B.interior[0] - other._a_B.interior[0]
        result._source.interior[0] = (
            self._source.interior[0] - other._source.interior[0]
        )
        return result

    @property
    def a_fx(self) -> Float[torch.Tensor, "C N N L"]:
        return self.a_P.face_harmonic_mean_along(Axis.X)[0]

    @property
    def a_fy(self) -> Float[torch.Tensor, "C N L N"]:
        return self.a_P.face_harmonic_mean_along(Axis.Y)[0]

    @property
    def a_fz(self) -> Float[torch.Tensor, "C L N N"]:
        return self.a_P.face_harmonic_mean_along(Axis.Z)[0]

    def apply(self, xi: CellField) -> Float[torch.Tensor, "C N N N"]:
        yi = self.a_P.interior[0] * xi.interior[0]
        yi += self.a_E.interior[0] * xi.get_shifted_interior_along(Axis.X, 1)[0]
        yi += (
            self.a_W.interior[0] * xi.get_shifted_interior_along(Axis.X, -1)[0]
        )
        yi += self.a_N.interior[0] * xi.get_shifted_interior_along(Axis.Y, 1)[0]
        yi += (
            self.a_S.interior[0] * xi.get_shifted_interior_along(Axis.Y, -1)[0]
        )
        yi += self.a_T.interior[0] * xi.get_shifted_interior_along(Axis.Z, 1)[0]
        yi += (
            self.a_B.interior[0] * xi.get_shifted_interior_along(Axis.Z, -1)[0]
        )
        return yi

    def get_coeff_along(self, axis: Axis, forward: bool) -> CellField:
        match axis:
            case Axis.X:
                return self.a_E if forward else self.a_W
            case Axis.Y:
                return self.a_N if forward else self.a_S
            case Axis.Z:
                return self.a_T if forward else self.a_B
            case _:
                raise ValueError(f"Invalid axis: {axis}")

    def set_coeff_along(
        self, axis: Axis, forward: bool, value: Float[torch.Tensor, "C N N N"]
    ) -> None:
        match axis:
            case Axis.X:
                if forward:
                    self.a_E.interior[0] = value
                else:
                    self.a_W.interior[0] = value
            case Axis.Y:
                if forward:
                    self.a_N.interior[0] = value
                else:
                    self.a_S.interior[0] = value
            case Axis.Z:
                if forward:
                    self.a_T.interior[0] = value
                else:
                    self.a_B.interior[0] = value
            case _:
                raise ValueError(f"Invalid axis: {axis}")
