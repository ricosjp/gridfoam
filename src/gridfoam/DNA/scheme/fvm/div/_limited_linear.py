import torch
from jaxtyping import Float

from gridfoam.DNA._grid._grid import PyOctreeNode, RawIndexConversionMode
from gridfoam.DNA.constants import DOMAIN_BOUNDARY_MAP, FACE_NEIGHBOR_MAP
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.enum import Axis, FieldLayout
from gridfoam.DNA.fielddata import CellField, FaceField, FVMatrix
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionType
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.div._interface import IFVMDivOperator
from gridfoam.DNA.scheme.tvd._limited_linear import LimitedLinear


class FVMDivLimitedLinear(IFVMDivOperator):
    def __init__(self, phi_fm: FieldMeta, psi_fm: FieldMeta) -> None:
        assert phi_fm.layout == FieldLayout.FACE
        assert psi_fm.layout == FieldLayout.CELL
        assert psi_fm.components == 1  # if C > 1, use LimitedLinearV
        self._phi_fm = phi_fm
        self._psi_fm = psi_fm
        self._limited_linear = LimitedLinear()

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
        V = torch.prod(dx)

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

        fvmatrix.source.interior[0] += self._compute_source(
            psi_c, phi_e, phi_w, phi_n, phi_s, phi_t, phi_b, V
        )

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
            if bc.target_field != self._psi_fm:
                continue
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
                    )
                    zero = torch.zeros_like(a_boundary)
                    source = fvmatrix.source.get_boundary_cell_along(
                        axis, forward
                    )
                    match bc.type:
                        case BoundaryConditionType.DIRICHLET:
                            # (C N N)
                            source -= a_boundary * bc.value[:, None, None]
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
                            source -= sign * a_boundary * 0.5 * dx * value
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

    def _compute_source(
        self,
        psi_c: CellField,
        phi_e: Float[torch.Tensor, "C N N N"],
        phi_w: Float[torch.Tensor, "C N N N"],
        phi_n: Float[torch.Tensor, "C N N N"],
        phi_s: Float[torch.Tensor, "C N N N"],
        phi_t: Float[torch.Tensor, "C N N N"],
        phi_b: Float[torch.Tensor, "C N N N"],
        V: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "C N N N"]:
        # (C N N N)
        zero = torch.zeros_like(phi_e)
        source = torch.zeros_like(zero)

        axis_config = [
            (Axis.X, (phi_e, phi_w)),
            (Axis.Y, (phi_n, phi_s)),
            (Axis.Z, (phi_t, phi_b)),
        ]

        for axis, (phi_f, phi_b) in axis_config:
            # Get all shifted values at once
            psi_m2 = psi_c.get_shifted_interior_along(axis, -2)[0]
            psi_m1 = psi_c.get_shifted_interior_along(axis, -1)[0]
            psi_0 = psi_c.get_shifted_interior_along(axis, 0)[0]
            psi_1 = psi_c.get_shifted_interior_along(axis, 1)[0]
            psi_2 = psi_c.get_shifted_interior_along(axis, 2)[0]

            # Pre-compute phi splits (reused for both forward and backward)
            phi_f_plus = torch.maximum(phi_f, zero)
            phi_f_minus = torch.minimum(phi_f, zero)
            phi_b_plus = torch.maximum(phi_b, zero)
            phi_b_minus = torch.minimum(phi_b, zero)

            # Pre-compute delta values (reused multiple times)
            delta_0_minus = psi_0 - psi_m1  # reused 3 times
            delta_0_plus = psi_1 - psi_0  # reused 3 times
            delta_1_plus = psi_2 - psi_1  # used once
            delta_m1_minus = psi_m1 - psi_m2  # used once

            # Forward face corrections
            correction_fwd1 = self._limited_linear.correction_term(
                delta_0_minus, delta_0_plus
            )
            source -= correction_fwd1 * phi_f_plus

            correction_fwd2 = self._limited_linear.correction_term(
                delta_0_plus, delta_1_plus
            )
            source += correction_fwd2 * phi_f_minus

            # Backward face corrections
            correction_bwd1 = self._limited_linear.correction_term(
                delta_m1_minus, delta_0_minus
            )
            source += correction_bwd1 * phi_b_plus

            correction_bwd2 = self._limited_linear.correction_term(
                delta_0_minus, delta_0_plus
            )
            source -= correction_bwd2 * phi_b_minus

        return source / V
