from gridfoam.core.field import CellField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.fv.schemes.ddt import ddt_coefficients


def ddt(field: CellField) -> FvMatrix:
    """
    Build the configured Euler or backward (BDF2) time-derivative term.

    Euler uses ``(psi - psi_old) * V / dt``. For equal time steps,
    backward uses ``(1.5*psi - 2*psi_old + 0.5*psi_older) * V / dt``.
    With insufficient history, backward starts with Euler. Volumes are
    assumed stationary between time levels.

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
    a, b, c = ddt_coefficients(field)
    mat.diag += a * vol_over_dt
    history = b * field.old_data
    if c != 0.0:
        assert field.older_data is not None
        history = history - c * field.older_data
    mat.source += vol_over_dt * history

    return mat
