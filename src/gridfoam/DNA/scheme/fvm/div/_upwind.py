import torch

from gridfoam.DNA._grid._grid import PyOctreeNode
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import BoundaryConditionType, FieldLayout
from gridfoam.DNA.fielddata import CellField, FaceField, FVMatrix
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.div._interface import IFVMDivOperator


class FVMDivUpwind(IFVMDivOperator):
    def __init__(self, phi_fm: FieldMeta, psi_fm: FieldMeta) -> None:
        assert phi_fm.layout == FieldLayout.FACE
        assert psi_fm.layout == FieldLayout.CELL
        self._phi_fm = phi_fm
        self._psi_fm = psi_fm

    def build(
        self,
        cube: PyOctreeNode,
        ctx: CtxForCubeOperation,
    ) -> FVMatrix:
        cube_field = cube.field
        dx = ctx.dx
        phi_f = cube_field.get_field(self._phi_fm)
        psi_c = cube_field.get_field(self._psi_fm)
        assert isinstance(phi_f, FaceField)
        assert isinstance(psi_c, CellField)
        C = psi_c.C
        N = psi_c.N
        H = psi_c.H
        dtype = psi_c.raw.dtype
        device = psi_c.raw.device
        fvmatrix = FVMatrix(C, N, H, dtype, device)
        V = dx[0] * dx[1] * dx[2]

        # (C N N N)
        phi_e = phi_f.x[0, :, :, :, 1:]
        phi_w = phi_f.x[0, :, :, :, :-1]
        phi_n = phi_f.y[0, :, :, 1:, :]
        phi_s = phi_f.y[0, :, :, :-1, :]
        phi_t = phi_f.z[0, :, 1:, :, :]
        phi_b = phi_f.z[0, :, :-1, :, :]

        zero = torch.zeros((C, N, N, N), dtype=dtype, device=device)

        # (C N N N)
        fvmatrix.a_E.interior[0] += torch.minimum(phi_e, zero) / V
        fvmatrix.a_W.interior[0] -= torch.maximum(phi_w, zero) / V
        fvmatrix.a_N.interior[0] += torch.minimum(phi_n, zero) / V
        fvmatrix.a_S.interior[0] -= torch.maximum(phi_s, zero) / V
        fvmatrix.a_T.interior[0] += torch.minimum(phi_t, zero) / V
        fvmatrix.a_B.interior[0] -= torch.maximum(phi_b, zero) / V

        # (C N N N)
        fvmatrix.a_P.interior[0] += (
            torch.maximum(phi_e, zero)
            - torch.minimum(phi_w, zero)
            + torch.maximum(phi_n, zero)
            - torch.minimum(phi_s, zero)
            + torch.maximum(phi_t, zero)
            - torch.minimum(phi_b, zero)
        ) / V
        return fvmatrix
