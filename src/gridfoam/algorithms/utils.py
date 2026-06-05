import logging

import torch

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.fv import fvc, fvm
from gridfoam.solvers.base import LinearSolver

logger = logging.getLogger(__name__)


def needs_reference_value(field: CellField) -> bool:
    """
    Check if the field needs a reference value to solve the equation.
    """
    for bc in field.bcs.values():
        if isinstance(bc, DirichletBC):
            return False
    return True


def set_reference_value(mat: FvMatrix):
    """
    Apply a pressure reference in OpenFOAM setReference style.
    """
    ref_cell = mat.field.ref_cell_id
    ref_diag = mat.diag[ref_cell].clone()
    mat.diag[ref_cell] += ref_diag
    mat.source[ref_cell] += ref_diag * mat.field.ref_value


def solve_pressure_poisson(
    p: CellField,
    rAU: CellField,
    phi: FaceField,
    solver: LinearSolver,
    *,
    p_needs_ref: bool,
    n_non_orthogonal_correctors: int,
) -> torch.Tensor:
    """
    Solve the pressure Poisson equation with non-orthogonal correctors.

    Parameters
    ----------
    p : CellField
        Pressure field.
    rAU : CellField
        Reciprocal momentum diagonal coefficient.
    phi : FaceField
        Face flux field supplying the continuity residual.
    solver : LinearSolver
        Linear solver for the pressure Poisson equation.
    p_needs_ref : bool
        Whether a reference pressure must be enforced.
    n_non_orthogonal_correctors : int
        Number of extra non-orthogonal correction passes.

    Returns
    -------
    torch.Tensor
        Solved pressure field data.
    """
    div_phi = fvc.div(phi).data
    p_solved = p.data

    for corr in range(n_non_orthogonal_correctors + 1):
        pEqn_mat = -fvm.laplacian(rAU.data, p)
        pEqn_mat.source = pEqn_mat.source - div_phi
        if p_needs_ref:
            set_reference_value(pEqn_mat)

        pressure_eq = equation("pressure_poisson", p, pEqn_mat)
        p_solved = solver.solve(pressure_eq)
        p.data = p_solved

        if corr < n_non_orthogonal_correctors:
            logger.debug(
                "non-orthogonal pressure corrector %d/%d",
                corr + 1,
                n_non_orthogonal_correctors,
            )

    return p_solved
