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
    lhs : FvMatrix
        Assembled left-hand-side matrix.
        The right-hand-side known term ``b`` is stored in ``lhs.source``.
    """

    def __init__(self, name: str, target: CellField, lhs: FvMatrix):
        self.name = name
        self.target = target
        self.lhs = lhs


def equation(name: str, target: CellField, lhs: FvMatrix) -> Equation:
    """
    Factory function to build an ``Equation`` instance.

    Parameters
    ----------
    name : str
        Equation name.
    target : CellField
        Target cell-centered field.
    lhs : FvMatrix
        Assembled discrete matrix.

    Returns
    -------
    Equation
        Constructed equation object.
    """
    return Equation(name, target, lhs)
