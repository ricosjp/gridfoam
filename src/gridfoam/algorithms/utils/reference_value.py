import logging

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix

logger = logging.getLogger(__name__)


def needs_reference_value(field: CellField) -> bool:
    """
    Check whether a field needs a reference value to solve the equation.

    A reference is required when no Dirichlet boundary condition is present,
    leaving the pressure (or similar) level undetermined.

    Parameters
    ----------
    field : CellField
        Field whose boundary conditions are inspected.

    Returns
    -------
    bool
        ``True`` when no Dirichlet boundary condition is configured.
    """
    for bc in field.bcs.values():
        if isinstance(bc, DirichletBC):
            return False
    return True


def set_reference_value(mat: FvMatrix, ref_cell: int, ref_value: float):
    """
    Apply a pressure reference in OpenFOAM ``setReference`` style.

    Doubles the diagonal entry at ``ref_cell`` and adds a matching source
    contribution so that the discrete system pins the field value there.

    Parameters
    ----------
    mat : FvMatrix
        Matrix equation to modify in place.
    ref_cell : int
        Cell index used as the reference location.
    ref_value : float
        Reference field value prescribed at ``ref_cell``.
    """
    ref_diag = mat.diag[ref_cell].clone()
    mat.diag[ref_cell] += ref_diag
    mat.source[ref_cell] += ref_diag * ref_value
