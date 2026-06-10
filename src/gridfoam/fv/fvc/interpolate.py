from gridfoam.core.field import CellField, FaceField, FieldRole
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.fvc.reconstruction import linear_internal_face_values


def interpolate(field: CellField) -> FaceField:
    """
    Linearly interpolate cell-centered values to face centers.

    When APIBM (Axis Projected Immersed Boundary Method) is enabled,
    boundary values are updated with immersed-boundary-aware states.

    Parameters
    ----------
    field : CellField
        Cell-centered field.

    Returns
    -------
    FaceField
        Interpolated face-centered field.
    """
    grid = field.grid

    psi_f = grid.get_field(f"{field.name}_f")
    if psi_f is None:
        psi_f = FaceField(
            grid,
            f"{field.name}_f",
            role=FieldRole.LOCAL,
            num_components=field.num_components,
            dimension=field.dimension,
            export=False,
        )
    assert isinstance(psi_f, FaceField)
    # Internal faces
    psi_f.single_data = linear_internal_face_values(field)

    for batch in iter_boundary_batches(field):
        _, _, _, psi_b = evaluate_boundary_state(field, batch)
        # Domain boundaries
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            psi_f.domain_bnd_data[batch.face_mask] = psi_b
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                psi_f.immersed_upper[batch.face_mask] = psi_b
                continue
            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                psi_f.immersed_lower[batch.face_mask] = psi_b
                continue

    return psi_f
