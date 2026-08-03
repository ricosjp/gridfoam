from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix


class Equation:
    """
    Container for equation metadata and assembled discrete matrix.

    Parameters
    ----------
    target : CellField
        Target cell-centered field to solve/update.
    fv_matrix : FvMatrix
        Assembled finite-volume matrix.
        The right-hand-side known term ``b`` is stored in ``fv_matrix.source``.

    Attributes
    ----------
    name : str
        Name of the field to be solved.
    target : CellField
        Target cell-centered field to solve/update.
    fv_matrix : FvMatrix
        Assembled finite-volume matrix.
    """

    name: str
    """Name of the field to be solved."""

    target: CellField
    """Target cell-centered field to solve/update."""

    fv_matrix: FvMatrix
    """Assembled finite-volume matrix."""

    def __init__(self, target: CellField, fv_matrix: FvMatrix):
        self.name = target.name
        self.target = target
        self.fv_matrix = fv_matrix


def equation(target: CellField, fv_matrix: FvMatrix) -> Equation:
    """
    Factory function to build an ``Equation`` instance.

    Parameters
    ----------
    target : CellField
        Target cell-centered field.
    fv_matrix : FvMatrix
        Assembled discrete matrix.

    Returns
    -------
    Equation
        Constructed equation object.
    """
    return Equation(target, fv_matrix)
