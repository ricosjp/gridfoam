import torch

from gridfoam.DNA._grid._grid import PyOctreeNode, RawIndexConversionMode
from gridfoam.DNA.constants import DOMAIN_BOUNDARY_MAP, FACE_NEIGHBOR_MAP
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.fielddata import CellField, FVMatrix
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionType
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
        ctx: CtxForCubeOperation,
    ) -> FVMatrix:
        cube_field = cube.field
        dx = ctx.dx
        gamma_c = cube_field.get_field(self._gamma_fm)
        psi_c = cube_field.get_field(self._psi_fm)
        assert isinstance(gamma_c, CellField)
        assert isinstance(psi_c, CellField)
        # This method supports only scalar fields currently.
        C = psi_c.C
        N = psi_c.N
        H = psi_c.H
        dtype = psi_c.raw.dtype
        device = psi_c.raw.device
        fvmatrix = FVMatrix(C, N, H, dtype, device)
        Sf = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        V = torch.prod(dx)
        gamma_f = gamma_c.face_harmonic_mean()

        # (C N N N)
        if gamma_c.C == 1:
            gamma_e = gamma_f.x[0, :, :, :, 1:]
            gamma_w = gamma_f.x[0, :, :, :, :-1]
            gamma_n = gamma_f.y[0, :, :, 1:, :]
            gamma_s = gamma_f.y[0, :, :, :-1, :]
            gamma_t = gamma_f.z[0, :, 1:, :, :]
            gamma_b = gamma_f.z[0, :, :-1, :, :]
        elif gamma_c.C == 3:
            gamma_e = gamma_f.x[0, 0, :, :, 1:]
            gamma_w = gamma_f.x[0, 0, :, :, :-1]
            gamma_n = gamma_f.y[0, 1, :, 1:, :]
            gamma_s = gamma_f.y[0, 1, :, :-1, :]
            gamma_t = gamma_f.z[0, 2, 1:, :, :]
            gamma_b = gamma_f.z[0, 2, :-1, :, :]
        else:
            raise ValueError(f"Invalid number of components: {gamma_c.C}")

        # (C N N N)
        a_E = gamma_e * Sf[0] / dx[0]
        a_W = gamma_w * Sf[0] / dx[0]
        a_N = gamma_n * Sf[1] / dx[1]
        a_S = gamma_s * Sf[1] / dx[1]
        a_T = gamma_t * Sf[2] / dx[2]
        a_B = gamma_b * Sf[2] / dx[2]
        a_P = -(a_E + a_W + a_N + a_S + a_T + a_B)
        fvmatrix.a_E.interior[0] += a_E / V
        fvmatrix.a_W.interior[0] += a_W / V
        fvmatrix.a_N.interior[0] += a_N / V
        fvmatrix.a_S.interior[0] += a_S / V
        fvmatrix.a_T.interior[0] += a_T / V
        fvmatrix.a_B.interior[0] += a_B / V
        fvmatrix.a_P.interior[0] += a_P / V

        self._apply_boundary_conditions(cube, ctx, fvmatrix)
        return fvmatrix

    def _apply_boundary_conditions(
        self,
        cube: PyOctreeNode,
        ctx: CtxForCubeOperation,
        fvmatrix: FVMatrix,
    ) -> None:
        nbr_codes = cube.cubecode.neighbor_codes(
            ctx.depth,
            ctx.bounds,
            RawIndexConversionMode.BORDER,
            False,
        )
        for bc in ctx.bcs:
            for label in bc.target_boundary_labels:
                if (
                    label in DOMAIN_BOUNDARY_MAP
                    and nbr_codes[DOMAIN_BOUNDARY_MAP[label]] is None
                ):
                    axis, forward = FACE_NEIGHBOR_MAP[
                        DOMAIN_BOUNDARY_MAP[label]
                    ]
                    target_coeff = fvmatrix.get_coeff_along(axis, forward)
                    a_boundary = target_coeff.get_boundary_cell_along(
                        axis, forward
                    )  # (T C N N)
                    zero = torch.zeros_like(a_boundary)
                    source = fvmatrix.source.get_boundary_cell_along(
                        axis, forward
                    )  # (T C N N)
                    match bc.type:
                        case BoundaryConditionType.DIRICHLET:
                            # (T C N N)
                            source -= (
                                2.0 * a_boundary * bc.value[None, :, None, None]
                            )
                            a_p_boundary = fvmatrix.a_P.get_boundary_cell_along(
                                axis, forward
                            )
                            fvmatrix.a_P.set_boundary_cell_along(
                                axis, forward, a_p_boundary - a_boundary
                            )
                            fvmatrix.source.set_boundary_cell_along(
                                axis, forward, source
                            )
                            target_coeff.set_boundary_cell_along(
                                axis, forward, zero
                            )
                        case BoundaryConditionType.NEUMANN:
                            sign = 1.0 if forward else -1.0
                            dx = ctx.dx[axis.value - 1]
                            value = bc.value[axis.value - 1]
                            source -= sign * a_boundary * dx * value
                            a_p_boundary = fvmatrix.a_P.get_boundary_cell_along(
                                axis, forward
                            )
                            fvmatrix.a_P.set_boundary_cell_along(
                                axis, forward, a_p_boundary + a_boundary
                            )
                            fvmatrix.source.set_boundary_cell_along(
                                axis, forward, source
                            )
                            target_coeff.set_boundary_cell_along(
                                axis, forward, zero
                            )
                        case _:
                            raise ValueError(
                                f"Invalid boundary condition type: {bc.type}"
                            )
