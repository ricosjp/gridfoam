"""Public face-interpolation entry point."""

from gridfoam.core.field import CellField, FaceField
from gridfoam.fv.schemes.interpolate import linear


def interpolate(field: CellField) -> FaceField:
    """
    Interpolate cell-centered values to face centers.

    Delegates to gridfoam ``linear``, which applies hierarchy-aware
    face-centre offset correction on hanging-node interfaces. Immersed
    faces are filled from boundary conditions rather than owner/neighbour
    interpolation.

    Parameters
    ----------
    field : CellField
        Cell-centered field.

    Returns
    -------
    FaceField
        Interpolated face-centered field named ``{field.name}_f``.
    """
    return linear(field)
