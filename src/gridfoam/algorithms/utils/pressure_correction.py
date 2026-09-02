"""Shared pressure-correction helpers for incompressible flow algorithms.

These functions implement the pressure--velocity coupling steps common to
SIMPLE, PISO, and PIMPLE after the momentum predictor solve.
"""

from __future__ import annotations

import logging

import torch
from jaxtyping import Float

from gridfoam.algorithms.utils import set_reference_value
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.fv import fvc, fvm
from gridfoam.fv.adjust_phi import adjust_phi
from gridfoam.fv.flux import correct_flux, set_phi_from_matrix_flux
from gridfoam.solvers.base import LinearSolver, SolveStats

logger = logging.getLogger(__name__)


def solve_pressure_poisson(
    p: CellField,
    rAU: CellField,
    div_phi: Float[torch.Tensor, " C 1"],
    solver: LinearSolver,
    *,
    n_non_orthogonal_correctors: int,
    p_needs_ref: bool,
    p_ref_cell: int | None = None,
    p_ref_value: float | None = None,
) -> tuple[FvMatrix, tuple[SolveStats, ...]]:
    """
    Solve the pressure Poisson equation with non-orthogonal correctors.

    Assembles and solves ``-laplacian(rAU, p) = div(phi)`` (negative
    Laplacian form) for each non-orthogonal corrector pass.

    Parameters
    ----------
    p : CellField
        Pressure field to solve.
    rAU : CellField
        Inverse momentum diagonal ``1 / A(U)``.
    div_phi : torch.Tensor
        Cell-centered divergence of the predicted face flux ``div(phi)``,
        normalized by the cell volume as returned by ``fvc.div``.
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

    Returns
    -------
    tuple[FvMatrix, tuple[SolveStats, ...]]
        Final pressure-equation matrix and per-component statistics from the
        last solve.
    """
    p_eqn_mat: FvMatrix | None = None
    last_stats: tuple[SolveStats, ...] | None = None
    # Explicit terms enter the matrix in volume-integrated form.
    div_phi_source = div_phi * p.grid.cell_volumes
    for corr in range(n_non_orthogonal_correctors + 1):
        p_eqn_mat = -fvm.laplacian(rAU.data, p)
        p_eqn_mat.source = p_eqn_mat.source - div_phi_source
        if p_needs_ref:
            if p_ref_cell is None or p_ref_value is None:
                raise ValueError(
                    "p_ref_cell and p_ref_value are required when "
                    "p_needs_ref is True"
                )
            set_reference_value(p_eqn_mat, p_ref_cell, p_ref_value)

        pressure_eq = equation(p, p_eqn_mat)
        solve_result = solver.solve(pressure_eq)
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
    return p_eqn_mat, last_stats


def correct_phi_inconsistent(
    phi: FaceField,
    rAU: CellField,
    p: CellField,
    phi_hbya: Float[torch.Tensor, " F_single 1"],
) -> None:
    """
    Correct face flux with the inconsistent pressure-gradient formulation.

    OpenFOAM default when ``consistent`` is disabled:

    ``phi = phi_HbyA - rAU_f * |Sf| * snGrad(p)``.

    Only single-sided internal faces in ``phi.single_data`` are updated;
    boundary fluxes are synchronized later by ``finalize_pressure_correction``.

    Parameters
    ----------
    phi : FaceField
        Face flux field to update.
    rAU : CellField
        Inverse momentum diagonal.
    p : CellField
        Pressure field used for the surface-normal gradient.
    phi_hbya : torch.Tensor
        Predicted face flux ``HbyA_f & Sf`` before pressure correction.
    """
    sn_grad_p = fvc.sn_grad(p)
    rAU_f = fvc.interpolate(rAU)
    mag_Sf = torch.linalg.vector_norm(
        phi.grid.Sf[phi.single_mask], dim=1, keepdim=True
    )
    set_phi_from_matrix_flux(
        phi,
        phi_hbya - rAU_f.single_data * mag_Sf * sn_grad_p.single_data,
    )


def correct_phi_from_pressure_equation(
    phi: FaceField,
    p_eqn_mat: FvMatrix,
    p: CellField,
    phi_hbya: Float[torch.Tensor, " F_single 1"],
) -> None:
    """
    Correct face flux using the solved pressure-equation matrix flux.

    OpenFOAM ``consistent`` formulation. The assembled matrix is
    ``-laplacian(rAU, p)``, so the diffusive flux uses the positive operator:

    ``phi = phi_HbyA - (-pEqn).flux(p)``.

    Parameters
    ----------
    phi : FaceField
        Face flux field to update.
    p_eqn_mat : FvMatrix
        Solved negative-Laplacian pressure equation matrix.
    p : CellField
        Pressure field passed to ``FvMatrix.flux``.
    phi_hbya : torch.Tensor
        Predicted face flux ``HbyA_f & Sf`` before pressure correction.
    """
    flux_p = (-p_eqn_mat).flux(p.data)
    set_phi_from_matrix_flux(phi, phi_hbya - flux_p)


def correct_phi(
    phi: FaceField,
    rAU: CellField,
    p: CellField,
    *,
    consistent: bool = False,
    phi_hbya: Float[torch.Tensor, " F_single 1"],
    p_eqn_mat: FvMatrix,
) -> None:
    """
    Apply pressure flux correction after the Poisson solve.

    Dispatches to the consistent matrix-flux path or the inconsistent
    ``snGrad(p)`` path according to ``consistent``.

    Parameters
    ----------
    phi : FaceField
        Face flux field to update.
    rAU : CellField
        Inverse momentum diagonal (used only when ``consistent=False``).
    p : CellField
        Pressure field used for correction.
    consistent : bool, optional
        If ``True``, use ``FvMatrix.flux`` (OpenFOAM ``consistent yes``).
        If ``False``, use ``rAU_f * |Sf| * snGrad(p)``.
    phi_hbya : torch.Tensor
        Predicted face flux ``HbyA_f & Sf`` before pressure correction.
    p_eqn_mat : FvMatrix
        Solved negative-Laplacian pressure equation matrix.
    """
    if consistent:
        correct_phi_from_pressure_equation(phi, p_eqn_mat, p, phi_hbya)
    else:
        correct_phi_inconsistent(phi, rAU, p, phi_hbya)


def correct_velocity(
    U: CellField,
    HbyA: CellField,
    rAU: CellField,
    p: CellField,
) -> None:
    """
    Correct cell-centered velocity after the pressure solve.

    ``U = HbyA - rAU * grad(p)``.

    Parameters
    ----------
    U : CellField
        Velocity field to update.
    HbyA : CellField
        Momentum predictor ``H(U) / A(U)``.
    rAU : CellField
        Inverse momentum diagonal.
    p : CellField
        Pressure field used for the gradient correction.
    """
    grad_p = fvc.grad(p)
    U.data = HbyA.data - rAU.data * grad_p.data


def finalize_pressure_correction(
    phi: FaceField,
    U: CellField,
    *,
    adjust_phi_enabled: bool = True,
) -> None:
    """
    Synchronize boundary fluxes and optionally apply ``adjustPhi``.

    Notes
    -----
    1. ``correct_flux(phi, U)`` rebuilds boundary face fluxes from ``U`` BCs.
    2. ``adjust_phi(phi, U)`` scales adjustable outlet fluxes for mass balance
       when enabled.

    Parameters
    ----------
    phi : FaceField
        Face flux field to synchronize.
    U : CellField
        Velocity field used for boundary flux reconstruction.
    adjust_phi_enabled : bool, optional
        Whether to call ``adjust_phi`` after boundary synchronization.
    """
    correct_flux(phi, U)
    if adjust_phi_enabled:
        adjust_phi(phi, U)
