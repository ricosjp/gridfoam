import torch
from jaxtyping import Float

from gridfoam.DNA._grid._grid import PyOctreeNode, RawIndexConversionMode
from gridfoam.DNA.constants import DOMAIN_BOUNDARY_MAP, FACE_NEIGHBOR_MAP
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import Axis, BoundaryConditionType, FieldLayout
from gridfoam.DNA.fielddata import FaceField
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta


class RhieChowCorrection:
    def __init__(
        self, U_fm: FieldMeta, p_fm: FieldMeta, momentum_eq: EquationMeta
    ) -> None:
        assert U_fm.layout == FieldLayout.CELL
        assert p_fm.layout == FieldLayout.CELL
        self._U_fm = U_fm
        self._p_fm = p_fm
        self._momentum_eq = momentum_eq

    def update_velocity(
        self,
        cube: PyOctreeNode,
        ctx: CtxForCubeOperation,
    ) -> None:
        cube_field = cube.field
        dx = ctx.dx
        U_c = cube_field.get_field(self._U_fm)
        p_c = cube_field.get_field(self._p_fm)
        fvmatrix = cube_field.get_fvmatrix(self._momentum_eq)
        gradp_c = cube_field.cells["gradp"]
        # Compute cell-centered pressure gradient
        # Use face-averaged pressure and compute gradient at cell center
        p_f = p_c.face_average()
        self._apply_dirichlet_boundary_conditions(cube, ctx, p_f)
        # x-direction: (T, C, N, N, L) -> (T, C, N, N, N) where L = N+1
        gradp_x = (p_f.x[0, 0, :, :, 1:] - p_f.x[0, 0, :, :, :-1]) / dx[0]
        # y-direction: (T, C, N, L, N) -> (T, C, N, N, N)
        gradp_y = (p_f.y[0, 0, :, 1:, :] - p_f.y[0, 0, :, :-1, :]) / dx[1]
        # z-direction: (T, C, L, N, N) -> (T, C, N, N, N)
        gradp_z = (p_f.z[0, 0, 1:, :, :] - p_f.z[0, 0, :-1, :, :]) / dx[2]
        gradp = torch.stack([gradp_x, gradp_y, gradp_z], dim=0)  # (3, N, N, N)
        self._apply_neumann_boundary_conditions(cube, ctx, gradp)
        gradp_c.interior[0] = gradp
        # (C, N, N, N) where C=3 for momentum
        ap = fvmatrix.a_P.interior[0]
        # Apply correction: U -= grad(p) / ap for each component
        U_c.interior[0] -= gradp / ap  # (3, N, N, N)

    def _apply_dirichlet_boundary_conditions(
        self, cube: PyOctreeNode, ctx: CtxForCubeOperation, p_f: FaceField
    ) -> None:
        nbr_codes = cube.cubecode.neighbor_codes(
            ctx.depth,
            ctx.bounds,
            RawIndexConversionMode.BORDER,
            False,
        )
        T = p_f.T
        C = p_f.C
        N = p_f.N
        for bc in ctx.bcs:
            if (
                bc.target_field != self._p_fm
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
                    p_f.set_boundary_face_along(axis, forward, values)

    def _apply_neumann_boundary_conditions(
        self,
        cube: PyOctreeNode,
        ctx: CtxForCubeOperation,
        gradp: Float[torch.Tensor, "3 N N N"],
    ) -> None:
        nbr_codes = cube.cubecode.neighbor_codes(
            ctx.depth,
            ctx.bounds,
            RawIndexConversionMode.BORDER,
            False,
        )
        N = gradp.shape[1]
        for bc in ctx.bcs:
            if (
                bc.target_field != self._p_fm
                or bc.type != BoundaryConditionType.NEUMANN
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
                    values = bc.value[:, None, None]
                    face = -1 if forward else 0
                    match axis:
                        case Axis.X:
                            gradp[:, :, :, face] = values
                        case Axis.Y:
                            gradp[:, :, face, :] = values
                        case Axis.Z:
                            gradp[:, face, :, :] = values
                        case _:
                            raise ValueError(f"Invalid axis: {axis}")
