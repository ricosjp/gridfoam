import torch

from gridfoam.DNA._grid._grid import PyOctreeNode
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
        cube: PyOctreeNode,
    ) -> FVMatrix:
        cube_field = cube.field
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
        V = dx[0] * dx[1] * dx[2]
        gamma_x = gamma_c.face_harmonic_mean_along(Axis.X) # (T C N N L)
        gamma_y = gamma_c.face_harmonic_mean_along(Axis.Y) # (T C N L N)
        gamma_z = gamma_c.face_harmonic_mean_along(Axis.Z) # (T C L N N)

        # (C N N N)
        gamma_e = gamma_x[0, :, :, :, 1:]
        gamma_w = gamma_x[0, :, :, :, :-1]
        gamma_n = gamma_y[0, :, :, 1:, :]
        gamma_s = gamma_y[0, :, :, :-1, :]
        gamma_t = gamma_z[0, :, 1:, :, :]
        gamma_b = gamma_z[0, :, :-1, :, :]

        # (C N N N)
        a_E = gamma_e * Sf[0]/ dx[0]
        a_W = gamma_w * Sf[0]/ dx[0]
        a_N = gamma_n * Sf[1]/ dx[1]
        a_S = gamma_s * Sf[1]/ dx[1]
        a_T = gamma_t * Sf[2]/ dx[2]
        a_B = gamma_b * Sf[2]/ dx[2]
        a_P = -(a_E + a_W + a_N + a_S + a_T + a_B)
        fvmatrix.a_E.interior[0] += a_E / V
        fvmatrix.a_W.interior[0] += a_W / V
        fvmatrix.a_N.interior[0] += a_N / V
        fvmatrix.a_S.interior[0] += a_S / V
        fvmatrix.a_T.interior[0] += a_T / V
        fvmatrix.a_B.interior[0] += a_B / V
        fvmatrix.a_P.interior[0] += a_P / V
        return fvmatrix
