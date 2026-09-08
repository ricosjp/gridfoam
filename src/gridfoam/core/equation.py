from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.fv.boundary_ops import immersed_dirichlet_constraints


class Equation:
    """
    Pair a target field with its constrained linear solve matrix.

    Apply near-boundary immersed Dirichlet cell constraints after all terms
    and explicit sources have been assembled. Keep the input matrix for
    physical face-flux reconstruction; elimination changes its couplings.

    Parameters
    ----------
    target : CellField
        Target cell-centered field to solve/update.
    fv_matrix : FvMatrix
        Complete assembled matrix, with the integrated RHS in ``source``.
        Construction leaves this input unchanged.

    Attributes
    ----------
    name : str
        Name of the field to be solved.
    target : CellField
        Target cell-centered field to solve/update.
    fv_matrix : FvMatrix
        Constrained solve matrix. Shares the input matrix when no cell
        constraints apply.
    """

    name: str
    """Name of the field to be solved."""

    target: CellField
    """Target cell-centered field to solve/update."""

    fv_matrix: FvMatrix
    """Solve matrix with immersed Dirichlet cell constraints applied."""

    def __init__(self, target: CellField, fv_matrix: FvMatrix):

        self.name = target.name
        self.target = target
        cells, values = immersed_dirichlet_constraints(target)
        self.fv_matrix = fv_matrix.with_fixed_values(cells, values)


def equation(target: CellField, fv_matrix: FvMatrix) -> Equation:
    """
    Build an equation and apply immersed Dirichlet cell constraints.

    Parameters
    ----------
    target : CellField
        Target cell-centered field.
    fv_matrix : FvMatrix
        Complete assembled matrix, including explicit sources. The input
        is left unchanged and remains suitable for face-flux reconstruction.

    Returns
    -------
    Equation
        Target field and constrained solve matrix. With no constraints,
        the solve matrix is the input matrix itself.
    """
    return Equation(target, fv_matrix)
