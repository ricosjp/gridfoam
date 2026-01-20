import torch

from gridfoam.DNA._grid._grid import PyOctreeNode, RawIndexConversionMode
from gridfoam.DNA.constants import DOMAIN_BOUNDARY_MAP, FACE_NEIGHBOR_MAP
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.fielddata import CellField, FaceField, FVMatrix
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionType
from gridfoam.DNA.meta.field import FieldMeta


class FVMDivLinear:
    def __init__(self, psi_fm: FieldMeta) -> None:
        assert psi_fm.layout == FieldLayout.CELL
        self._psi_fm = psi_fm

    def build(
        self,
        cube: PyOctreeNode,
        ctx: CtxForCubeOperation,
    ) -> FVMatrix:
        cube_field = cube.field
        dx = ctx.dx
        psi_c = cube_field.get_field(self._psi_fm)
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        V = torch.prod(dx)
        assert isinstance(psi_c, CellField)
        N = psi_c.N
        H = psi_c.H
        dtype = psi_c.raw.dtype
        device = psi_c.raw.device
        fvmatrix = FVMatrix(1, N, H, dtype, device)

        psi_f = psi_c.face_average()
        self._apply_dirichlet_boundary_conditions(cube, ctx, psi_f)
        psi_f_diag = FaceField(
            psi_c.T, 1, psi_c.N, psi_c.raw.dtype, psi_c.raw.device
        )
        psi_f_diag.x = psi_f.x[:, [0], :, :, :]  # (T 1 N N L)
        psi_f_diag.y = psi_f.y[:, [1], :, :, :]  # (T 1 N L N)
        psi_f_diag.z = psi_f.z[:, [2], :, :, :]  # (T 1 L N N)
        div_result = psi_f_diag.integrate_dSn(Sf) / V  # (T 1 N N N)
        fvmatrix.source = -div_result[0]
        return fvmatrix

    def _apply_dirichlet_boundary_conditions(
        self, cube: PyOctreeNode, ctx: CtxForCubeOperation, psi_f: FaceField
    ) -> None:
        """Apply Dirichlet boundary conditions to face values."""
        nbr_codes = cube.cubecode.neighbor_codes(
            ctx.depth,
            ctx.bounds,
            RawIndexConversionMode.BORDER,
            False,
        )
        T = psi_f.T
        C = psi_f.C
        N = psi_f.N
        for bc in ctx.bcs:
            if (
                bc.target_field != self._psi_fm
                or bc.type != BoundaryConditionType.DIRICHLET
            ):
                continue
            for label in bc.target_boundary_labels:
                if (
                    label in DOMAIN_BOUNDARY_MAP
                    and nbr_codes[DOMAIN_BOUNDARY_MAP[label]] is None
                ):
                    axis, forward = FACE_NEIGHBOR_MAP[
                        DOMAIN_BOUNDARY_MAP[label]
                    ]
                    values = bc.value[None, :, None, None].expand(T, C, N, N)
                    psi_f.set_boundary_face_along(axis, forward, values)
