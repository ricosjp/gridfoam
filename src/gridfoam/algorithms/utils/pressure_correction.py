"""Shared pressure-correction helpers for incompressible flow algorithms.

Common SIMPLE / PISO / PIMPLE steps after the momentum predictor
(OpenFOAM ``pEqn.H`` structure):

1. ``phiHbyA = flux(HbyA) [+ rAU_f * ddtCorr]``
   (:func:`~gridfoam.fv.flux.compute_phi_hbya`).
2. ``adjustPhi(phiHbyA, U, p)``.
3. Optional SIMPLEC with ``rAtU`` (:func:`apply_simplec`).
4. ``laplacian(rAtU, p) == div(phiHbyA)`` (:func:`solve_pressure_poisson`).
5. ``phi = phiHbyA - pEqn.flux()`` (:func:`correct_phi`).
6. ``U = HbyA - rAtU * grad(p)`` (:func:`correct_velocity`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from jaxtyping import Float

from gridfoam.algorithms.utils import set_reference_value
from gridfoam.algorithms.utils.residual import field_initial_residual
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv import fvc, fvm
from gridfoam.fv.boundary_ops import (
    apply_immersed_dirichlet_values,
    boundary_block,
    boundary_fixed_value_mask,
    boundary_normal_gradient,
    iter_boundary_states,
)
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.solvers.base import LinearSolver, SolveStats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PressureSolveResult:
    """
    Outcome of :func:`solve_pressure_poisson`.

    Attributes
    ----------
    matrix : FvMatrix
        Pressure-equation matrix of the last non-orthogonal pass
        (``-laplacian(rAtU, p)`` with the divergence source applied), before
        immersed cell-constraint elimination. Use it for flux correction.
    stats : tuple[SolveStats, ...]
        Linear-solver statistics of the last pass.
    initial_residual : float
        Residual before the first pressure solve (PIMPLE baseline, SIMPLE).
    last_initial_residual : float
        Residual before the last non-orthogonal solve (PIMPLE current).
    """

    matrix: FvMatrix
    stats: tuple[SolveStats, ...]
    initial_residual: float
    last_initial_residual: float


def solve_pressure_poisson(
    p: CellField,
    rAtU: CellField,
    div_phi_hbya: Float[torch.Tensor, " C"],
    solver: LinearSolver,
    *,
    n_non_orthogonal_correctors: int,
    p_needs_ref: bool,
    p_ref_cell: int | None = None,
    p_ref_value: float | None = None,
    final_solver: LinearSolver | None = None,
) -> PressureSolveResult:
    """
    Solve the pressure Poisson equation with non-orthogonal correctors.

    Assembles and solves ``-laplacian(rAtU, p) = -div(phiHbyA)`` for each
    non-orthogonal corrector pass.

    Parameters
    ----------
    p : CellField
        Pressure field to solve.
    rAtU : CellField
        Diffusion coefficient of the pressure equation: ``1 / A(U)`` or the
        SIMPLEC coefficient from :func:`simplec_rAtU`.
    div_phi_hbya : torch.Tensor
        Cell-centered divergence of the predicted face flux, normalized by
        the cell volume as returned by ``fvc.div``.
    solver : LinearSolver
        Linear solver used for the pressure equation.
    n_non_orthogonal_correctors : int
        Number of extra non-orthogonal pressure passes after the first solve.
    p_needs_ref : bool
        Whether a pressure reference value must be applied.
    p_ref_cell : int | None, optional
        Reference cell index for pressure.
    p_ref_value : float | None, optional
        Reference pressure value.
    final_solver : LinearSolver | None, optional
        Solver for the last non-orthogonal pass. When omitted, every pass
        uses ``solver`` (SIMPLE). PISO/PIMPLE pass ``pFinal`` here on the
        last inner corrector.

    Returns
    -------
    PressureSolveResult
        Final matrix, solver statistics, and first/last initial residuals.
    """
    p_eqn_mat: FvMatrix | None = None
    last_stats: tuple[SolveStats, ...] | None = None
    initial_residual: float | None = None
    last_initial_residual: float | None = None
    # Explicit terms enter the matrix in volume-integrated form.
    div_source = div_phi_hbya * p.grid.cell_volumes
    for corr in range(n_non_orthogonal_correctors + 1):
        p_eqn_mat = -fvm.laplacian(rAtU.data, p)
        p_eqn_mat.source = p_eqn_mat.source - div_source
        if p_needs_ref:
            if p_ref_cell is None or p_ref_value is None:
                raise ValueError(
                    "p_ref_cell and p_ref_value are required when "
                    "p_needs_ref is True"
                )
            set_reference_value(p_eqn_mat, p_ref_cell, p_ref_value)

        pressure_eq = equation(p, p_eqn_mat)
        last_initial_residual = field_initial_residual(pressure_eq.fv_matrix, p)
        if initial_residual is None:
            initial_residual = last_initial_residual

        pass_solver = (
            final_solver
            if final_solver is not None and corr == n_non_orthogonal_correctors
            else solver
        )
        solve_result = pass_solver.solve(pressure_eq)
        p.data = solve_result.solution
        last_stats = solve_result.stats

        if corr < n_non_orthogonal_correctors:
            logger.debug(
                "non-orthogonal pressure corrector %d/%d",
                corr + 1,
                n_non_orthogonal_correctors,
            )

    assert p_eqn_mat is not None
    assert last_stats is not None
    assert initial_residual is not None
    assert last_initial_residual is not None
    return PressureSolveResult(
        p_eqn_mat, last_stats, initial_residual, last_initial_residual
    )


def simplec_rAtU(
    UEqn_mat: FvMatrix,
    rAU: CellField,
    *,
    bounded: bool = True,
) -> Float[torch.Tensor, " C"]:
    """
    SIMPLEC pressure-equation coefficient ``rAtU = 1 / (1/rAU - H1)``.

    Parameters
    ----------
    UEqn_mat : FvMatrix
        Assembled (and relaxed) momentum matrix.
    rAU : CellField
        Inverse momentum diagonal ``1 / A(U)``.
    bounded : bool, default True
        If True (default), apply the OpenFOAM ``pimpleFoam`` safeguard
        ``1 / max(1/rAU - H1, 0.1/rAU)``. On octree / immersed-boundary
        meshes ``1/rAU - H1`` can lose positivity, which would make
        ``rAtU`` negative and break the pressure CG solve. Set ``False``
        only to match the unbounded ``simpleFoam`` formula on well-behaved
        meshes.

    Returns
    -------
    torch.Tensor
        ``rAtU`` cell values with shape ``[C]``.
    """
    inv = 1.0 / rAU.data - UEqn_mat.H1()
    if bounded:
        inv = torch.maximum(inv, 0.1 / rAU.data)
    return 1.0 / inv


def _add_boundary_sn_grad_flux(
    face_field: FaceField,
    p: CellField,
    cell_coeff: Float[torch.Tensor, " C"],
    *,
    sign: float,
) -> None:
    """
    Add ``sign * coeff_P * |Sf| * snGrad(p)_b`` on every pressure boundary.
    """
    for state in iter_boundary_states(p):
        batch = state.batch
        mag_Sf = batch.mag_Sf
        correction = (
            sign
            * cell_coeff[batch.target_cells]
            * mag_Sf
            * boundary_normal_gradient(p, state)
        )
        block = boundary_block(face_field, batch.face_kind)
        block[batch.face_mask] = block[batch.face_mask] + correction


def _copy_boundary_flux(dst: FaceField, src: FaceField) -> None:
    """Copy domain and immersed boundary flux blocks from ``src`` to ``dst``."""
    dst.domain_bnd_data = src.domain_bnd_data.clone()
    grid = dst.grid
    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        dst.immersed_upper = src.immersed_upper.clone()
        dst.immersed_lower = src.immersed_lower.clone()


def apply_simplec(
    phi_hbya: FaceField,
    HbyA: CellField,
    p: CellField,
    rAU: CellField,
    rAtU: CellField,
) -> None:
    """
    Apply the SIMPLEC (``consistent``) corrections to ``phiHbyA``/``HbyA``.

    ``phiHbyA += interpolate(rAtU - rAU) * snGrad(p) * |Sf|`` and
    ``HbyA -= (rAU - rAtU) * grad(p)``, as in OpenFOAM ``simpleFoam`` /
    ``pimpleFoam`` with ``consistent yes``. Boundary faces use the boundary
    normal gradient implied by the pressure boundary condition.

    Parameters
    ----------
    phi_hbya : FaceField
        Predicted flux, updated in place on all blocks.
    HbyA : CellField
        Momentum predictor, updated in place.
    p : CellField
        Current pressure.
    rAU : CellField
        ``1 / A(U)``.
    rAtU : CellField
        SIMPLEC coefficient from :func:`simplec_rAtU`.
    """
    geo = face_geometry(p.grid)
    d_r = rAtU.data - rAU.data
    d_r_f = geo.w_s * d_r[geo.owner_s] + (1.0 - geo.w_s) * d_r[geo.neighbour_s]
    sn_grad_p = fvc.sn_grad(p).single_data
    phi_hbya.single_data = (
        phi_hbya.single_data + d_r_f * geo.mag_Sf_s * sn_grad_p
    )
    _add_boundary_sn_grad_flux(phi_hbya, p, d_r, sign=1.0)

    grad_p = fvc.grad(p)
    HbyA.data = HbyA.data - (rAU.data - rAtU.data)[:, None] * grad_p.data


def correct_phi(
    phi: FaceField,
    phi_hbya: FaceField,
    p_eqn_mat: FvMatrix,
    p: CellField,
    rAtU: CellField,
) -> None:
    """
    Set ``phi = phiHbyA - pEqn.flux()`` on internal and boundary faces.

    Internal faces use :meth:`~gridfoam.core.fvmatrix.FvMatrix.flux`
    (including hanging-node correction). Boundary faces use
    ``rAtU_P |Sf| (p_b - p_P) / |d|``, which is zero on zero-gradient
    patches. On snapped Dirichlet cells, recover boundary flux from the
    cell mass balance. The result is divergence-free to solver tolerance.

    Parameters
    ----------
    phi : FaceField
        Face flux field to update.
    phi_hbya : FaceField
        Predicted flux from ``compute_phi_hbya`` (after ``adjust_phi`` /
        ``apply_simplec`` when used).
    p_eqn_mat : FvMatrix
        Assembled negative-Laplacian pressure matrix before cell-constraint
        elimination, including explicit face-flux corrections.
    p : CellField
        Solved pressure.
    rAtU : CellField
        Diffusion coefficient of the pressure equation.
    """
    flux_p = (-p_eqn_mat).flux(p.data)
    phi.single_data = phi_hbya.single_data - flux_p
    _copy_boundary_flux(phi, phi_hbya)
    _add_boundary_sn_grad_flux(phi, p, rAtU.data, sign=-1.0)
    _close_fixed_cell_flux(phi, p)


def _close_fixed_cell_flux(phi: FaceField, p: CellField) -> None:
    """Recover the undetermined Dirichlet flux on cells fixed by snapping.

    Their Poisson rows have been replaced by p_P = p_b. The original cell
    balance determines the total boundary flux instead. Split that flux
    over near Dirichlet faces by area; other cells/faces are unchanged.
    """
    selected = []
    area_sum = torch.zeros_like(p.grid.cell_volumes)
    for state in iter_boundary_states(p):
        batch = state.batch
        fixed = boundary_fixed_value_mask(p, batch, state.fraction)
        if not bool(torch.any(fixed)):
            continue
        area = batch.mag_Sf
        area = torch.where(fixed, area, torch.zeros_like(area))
        area_sum.index_add_(0, batch.target_cells, area)
        selected.append((batch, area))
    if not selected:
        return
    imbalance = fvc.div(phi).data * p.grid.cell_volumes
    for batch, area in selected:
        total = area_sum[batch.target_cells]
        fraction = area / torch.where(total > 0, total, torch.ones_like(total))
        block = boundary_block(phi, batch.face_kind)
        block[batch.face_mask] = (
            block[batch.face_mask] - fraction * imbalance[batch.target_cells]
        )


def correct_velocity(
    U: CellField,
    HbyA: CellField,
    rAtU: CellField,
    p: CellField,
) -> None:
    """
    Correct cell-centered velocity after the pressure solve.

    Apply ``U = HbyA - rAtU * grad(p)``, then restore prescribed values on
    snapped immersed Dirichlet cells.

    Parameters
    ----------
    U : CellField
        Velocity field to update.
    HbyA : CellField
        Momentum predictor ``H(U) / A(U)`` (SIMPLEC-adjusted if enabled).
    rAtU : CellField
        Diffusion coefficient used in the pressure equation.
    p : CellField
        Pressure field used for the gradient correction.
    """
    grad_p = fvc.grad(p)
    U.data = HbyA.data - rAtU.data[:, None] * grad_p.data
    apply_immersed_dirichlet_values(U)
