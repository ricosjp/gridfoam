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


class FVMDivLimitedLinearV(IFVMDivOperator):
    def __init__(self, phi_fm: FieldMeta, psi_fm: FieldMeta) -> None:
        assert phi_fm.layout == FieldLayout.FACE
        assert psi_fm.layout == FieldLayout.CELL
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
        # V scheme: compute single limiter based on steepest gradient direction
        # and apply to all components
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
            # (C, N, N, N)
            psi_m2 = psi_c.get_shifted_interior_along(axis, -2)[0]
            psi_m1 = psi_c.get_shifted_interior_along(axis, -1)[0]
            psi_0 = psi_c.get_shifted_interior_along(axis, 0)[0]
            psi_1 = psi_c.get_shifted_interior_along(axis, 1)[0]
            psi_2 = psi_c.get_shifted_interior_along(axis, 2)[0]

            # Pre-compute phi splits
            phi_f_plus = torch.maximum(phi_f, zero)
            phi_f_minus = torch.minimum(phi_f, zero)
            phi_b_plus = torch.maximum(phi_b, zero)
            phi_b_minus = torch.minimum(phi_b, zero)

            # V scheme: compute vector magnitude differences
            # and compute single limiter based on magnitude
            delta_0_minus_vec = psi_0 - psi_m1  # (C, N, N, N)
            delta_0_plus_vec = psi_1 - psi_0  # (C, N, N, N)
            delta_1_plus_vec = psi_2 - psi_1  # (C, N, N, N)
            delta_m1_minus_vec = psi_m1 - psi_m2  # (C, N, N, N)

            # Compute magnitudes (L2 norm along component dimension)
            # (C, N, N, N) -> (N, N, N)
            delta_0_minus_mag = torch.norm(delta_0_minus_vec, dim=0)
            delta_0_plus_mag = torch.norm(delta_0_plus_vec, dim=0)
            delta_1_plus_mag = torch.norm(delta_1_plus_vec, dim=0)
            delta_m1_minus_mag = torch.norm(delta_m1_minus_vec, dim=0)

            # Compute single limiter based on magnitude
            # (N, N, N)
            correction_fwd1_mag = self._limited_linear.correction_term(
                delta_0_minus_mag, delta_0_plus_mag
            )
            correction_fwd2_mag = self._limited_linear.correction_term(
                delta_0_plus_mag, delta_1_plus_mag
            )
            correction_bwd1_mag = self._limited_linear.correction_term(
                delta_m1_minus_mag, delta_0_minus_mag
            )
            correction_bwd2_mag = self._limited_linear.correction_term(
                delta_0_minus_mag, delta_0_plus_mag
            )

            # Expand limiter to all components: (N, N, N) -> (C, N, N, N)
            correction_fwd1 = correction_fwd1_mag[None, :, :, :].expand_as(
                phi_f
            )
            correction_fwd2 = correction_fwd2_mag[None, :, :, :].expand_as(
                phi_f
            )
            correction_bwd1 = correction_bwd1_mag[None, :, :, :].expand_as(
                phi_b
            )
            correction_bwd2 = correction_bwd2_mag[None, :, :, :].expand_as(
                phi_b
            )

            # Apply corrections (same limiter applied to all components)
            source -= correction_fwd1 * phi_f_plus
            source += correction_fwd2 * phi_f_minus
            source += correction_bwd1 * phi_b_plus
            source -= correction_bwd2 * phi_b_minus

        return source / V
