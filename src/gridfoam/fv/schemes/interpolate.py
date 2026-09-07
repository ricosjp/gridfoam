"""Face-interpolation schemes."""

from __future__ import annotations

from gridfoam.core.field import (
    CellField,
    FaceField,
    FieldRole,
    get_or_create_facefield,
)
from gridfoam.fv.boundary_ops import fill_boundary_face_values
from gridfoam.fv.kernels.face_geometry import face_geometry
from gridfoam.fv.kernels.face_interpolation import (
    correct_internal_values,
    linear_internal_face_values,
)
from gridfoam.fv.kernels.least_squares import least_squares_gradient


def linear(field: CellField) -> FaceField:
    """
    Interpolate a cell field to faces with hierarchy-aware linear weights.

    Gridfoam ``linear`` includes the face-centre offset correction used for
    hanging-node interfaces. On an octree the offset is non-zero only on
    2:1 faces, so the correction is evaluated on that subset using a local
    weighted least-squares gradient of the cells adjacent to those faces.
    The least-squares gradient is exact for linear fields, which keeps the
    corrected face values (and any Green-Gauss gradient built from them)
    exact for linear fields across refinement interfaces. Boundary faces
    without a boundary condition are filled by zero-gradient extrapolation.

    Parameters
    ----------
    field : CellField
        Cell-centered field.

    Returns
    -------
    FaceField
        Face-centered field named ``{field.name}_f``.
    """
    grid = field.grid
    geo = face_geometry(grid)
    psi_f = get_or_create_facefield(
        grid,
        f"{field.name}_f",
        FieldRole.LOCAL,
        field.component_shape,
        dimension=field.dimension,
    )

    base_values = linear_internal_face_values(field, geo)
    fill_boundary_face_values(field, psi_f)

    if geo.num_hanging > 0:
        grad_hang = least_squares_gradient(field, geo, hanging_cells_only=True)
        psi_f.single_data = correct_internal_values(
            field, base_values, grad_hang, geo
        )
    else:
        psi_f.single_data = base_values
    return psi_f
