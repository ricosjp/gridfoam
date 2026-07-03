import logging

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix

logger = logging.getLogger(__name__)


def needs_reference_value(field: CellField) -> bool:
    """
    Check if the field needs a reference value to solve the equation.
    """
    for bc in field.bcs.values():
        if isinstance(bc, DirichletBC):
            return False
    return True


def set_reference_value(mat: FvMatrix, ref_cell: int, ref_value: float):
    """
    Apply a pressure reference in OpenFOAM setReference style.
    """
    ref_diag = mat.diag[ref_cell].clone()
    mat.diag[ref_cell] += ref_diag
    mat.source[ref_cell] += ref_diag * ref_value
