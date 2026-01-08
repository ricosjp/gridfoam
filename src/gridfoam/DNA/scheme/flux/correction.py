import torch

from gridfoam.DNA._grid._grid import PyOctreeNode, RawIndexConversionMode
from gridfoam.DNA.constants import DOMAIN_BOUNDARY_MAP, FACE_NEIGHBOR_MAP
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.fielddata import FaceField
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionType
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta


class FluxCorrection:
    def __init__(
        self, phi_fm: FieldMeta, U_fm: FieldMeta, momentum_eq: EquationMeta
    ) -> None:
        assert phi_fm.layout == FieldLayout.FACE
        assert U_fm.layout == FieldLayout.CELL
        assert momentum_eq.target_field == U_fm
        self._phi_fm = phi_fm
        self._U_fm = U_fm
        self._momentum_eq = momentum_eq

    def update_flux(
        self,
        cube: PyOctreeNode,
        ctx: CtxForCubeOperation,
    ) -> None:
        cube_field = cube.field
        dx = ctx.dx
        U_c = cube_field.get_field(self._U_fm)
        phi_f = cube_field.get_field(self._phi_fm)
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        Uf = U_c.face_average()
        self._apply_dirichlet_boundary_conditions(cube, ctx, Uf)
        phi_f.x[0] = Uf.x[0, 0] * Sf[0]
        phi_f.y[0] = Uf.y[0, 1] * Sf[1]
        phi_f.z[0] = Uf.z[0, 2] * Sf[2]

    def _apply_dirichlet_boundary_conditions(
        self, cube: PyOctreeNode, ctx: CtxForCubeOperation, Uf: FaceField
    ) -> None:
        nbr_codes = cube.cubecode.neighbor_codes(
            ctx.depth,
            ctx.bounds,
            RawIndexConversionMode.BORDER,
            False,
        )
        T = Uf.T
        C = Uf.C
        N = Uf.N
        for bc in ctx.bcs:
            if bc.target_field != self._U_fm or bc.type != BoundaryConditionType.DIRICHLET:
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
                    Uf.set_boundary_face_along(axis, forward, values)
