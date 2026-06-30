from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix


class Equation:
    """
    Container for equation metadata and assembled discrete matrix.

    Parameters
    ----------
    name : str
        Equation name (e.g., ``"momentum"``, ``"pressure"``).
    target : CellField
        Target cell-centered field to solve/update.
    fv_matrix : FvMatrix
        Assembled finite-volume matrix.
        The right-hand-side known term ``b`` is stored in ``fv_matrix.source``.
    """

    def __init__(self, name: str, target: CellField, fv_matrix: FvMatrix):
        self.name = name
        self.target = target
        self.fv_matrix = fv_matrix


def equation(name: str, target: CellField, fv_matrix: FvMatrix) -> Equation:
    """
    Factory function to build an ``Equation`` instance.

    Parameters
    ----------
    name : str
        Equation name.
    target : CellField
        Target cell-centered field.
    fv_matrix : FvMatrix
        Assembled discrete matrix.

    Returns
    -------
    Equation
        Constructed equation object.
    """
    return Equation(name, target, fv_matrix)
