import torch

from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.fielddata import FVMatrix
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.ddt._interface import IFVMDdtOperator


class FVMDdtEuler(IFVMDdtOperator):
    def __init__(self, psi_fm: FieldMeta) -> None:
        assert psi_fm.layout == FieldLayout.CELL
        self._psi_fm = psi_fm

    def build(
        self,
        cube_field: CubeField,
    ) -> FVMatrix:
        dt = cube_field.dt
        psi = cube_field.get_field(self._psi_fm)
        C = psi.C
        N = psi.N
        H = psi.H
        dtype = psi.raw.dtype
        device = psi.raw.device
        fvmatrix = FVMatrix(C, N, H, dtype, device)
        rdt = 1.0 / dt
        fvmatrix.a_P = torch.full_like(psi.interior[0], rdt) # (C N N N)
        fvmatrix.source = rdt * psi.interior[0] # (C N N N)
        return fvmatrix
