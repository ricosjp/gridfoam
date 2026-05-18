from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix


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
