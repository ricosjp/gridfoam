from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix


def ddt(field: CellField) -> FvMatrix:
    """
    Build implicit Euler time-derivative term.

    ddt(psi) ~= (psi - psi_old) * V / dt.

    Parameters
    ----------
    field : CellField
        Target cell-centered field.

    Returns
    -------
    FvMatrix
        Matrix/source contribution for the transient term.
    """
    mat = FvMatrix(field)
    dt = field.grid.dt
    V = field.grid.cell_volumes

    vol_over_dt = V / dt
    mat.diag += vol_over_dt
    mat.source += vol_over_dt * field.old_data

    return mat
