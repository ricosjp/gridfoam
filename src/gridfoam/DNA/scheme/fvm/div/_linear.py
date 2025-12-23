
from gridfoam.DNA._grid._grid import PyOctreeNode
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.fielddata import CellField, FVMatrix
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvc.div._linear import FVCDivLinear


class FVMDivLinear:
    def __init__(self, psi_fm: FieldMeta) -> None:
        assert psi_fm.layout == FieldLayout.CELL
        self._psi_fm = psi_fm

    def build(
        self,
        cube: PyOctreeNode,
    ) -> FVMatrix:
        cube_field = cube.field
        dx = cube_field.dx
        psi_c = cube_field.get_field(self._psi_fm)
        assert isinstance(psi_c, CellField)
        C = psi_c.C
        N = psi_c.N
        H = psi_c.H
        dtype = psi_c.raw.dtype
        device = psi_c.raw.device
        fvmatrix = FVMatrix(C, N, H, dtype, device)
        fvmatrix.source = -FVCDivLinear().apply(psi_c, dx)[0]
        return fvmatrix
