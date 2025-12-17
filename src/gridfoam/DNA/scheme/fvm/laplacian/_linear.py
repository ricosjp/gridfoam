import torch

from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.enum import Axis, FieldLayout
from gridfoam.DNA.fielddata import CellField, FVMatrix
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.laplacian._interface import IFVMLaplacianOperator


class FVMLaplacianLinear(IFVMLaplacianOperator):
    def __init__(self, gamma_fm: FieldMeta, psi_fm: FieldMeta) -> None:
        assert gamma_fm.layout == FieldLayout.CELL
        assert psi_fm.layout == FieldLayout.CELL
        self._gamma_fm = gamma_fm
        self._psi_fm = psi_fm

    def build(
        self,
        cube_field: CubeField,
    ) -> FVMatrix:
        dx = cube_field.dx
        gamma_c = cube_field.get_field(self._gamma_fm)
        psi_c = cube_field.get_field(self._psi_fm)
        assert isinstance(gamma_c, CellField)
        assert isinstance(psi_c, CellField)
        # This method supports only scalar fields currently.
        assert gamma_c.C == 1
        assert psi_c.C == 1
        C = psi_c.C
        N = psi_c.N
        H = psi_c.H
        dtype = psi_c.raw.dtype
        device = psi_c.raw.device
        fvmatrix = FVMatrix(C, N, H, dtype, device)
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])

        # (C N N N)
        gamma_P = gamma_c.interior[0]
        gamma_E = gamma_c.get_shifted_interior_along(Axis.X, 1)[0] # (C N N N)
        gamma_W = gamma_c.get_shifted_interior_along(Axis.X, -1)[0] # (C N N N)
        gamma_N = gamma_c.get_shifted_interior_along(Axis.Y, 1)[0] # (C N N N)
        gamma_S = gamma_c.get_shifted_interior_along(Axis.Y, -1)[0] # (C N N N)
        gamma_T = gamma_c.get_shifted_interior_along(Axis.Z, 1)[0] # (C N N N)
        gamma_B = gamma_c.get_shifted_interior_along(Axis.Z, -1)[0] # (C N N N)

        # (C N N N)
        a_E = Sf[0] * 0.5 *(gamma_E - gamma_P) / dx[0]
        a_W = Sf[0] * 0.5 *(gamma_W - gamma_P) / dx[0]
        a_N = Sf[1] * 0.5 *(gamma_N - gamma_P) / dx[1]
        a_S = Sf[1] * 0.5 *(gamma_S - gamma_P) / dx[1]
        a_T = Sf[2] * 0.5 *(gamma_T - gamma_P) / dx[2]
        a_B = Sf[2] * 0.5 *(gamma_B - gamma_P) / dx[2]
        a_P = -(a_E + a_W + a_N + a_S + a_T + a_B)
        fvmatrix.a_E.interior[0] += a_E
        fvmatrix.a_W.interior[0] += a_W
        fvmatrix.a_N.interior[0] += a_N
        fvmatrix.a_S.interior[0] += a_S
        fvmatrix.a_T.interior[0] += a_T
        fvmatrix.a_B.interior[0] += a_B
        fvmatrix.a_P.interior[0] += a_P
        return fvmatrix
