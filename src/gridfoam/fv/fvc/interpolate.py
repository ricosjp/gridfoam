import torch

from gridfoam.core.field import CellField, FaceField, FieldRole
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)


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
    owner_single = grid.owner[psi_f.single_mask]
    neighbour_single = grid.neighbour[psi_f.single_mask]

    # Internal faces
    axis_single = grid.axis[psi_f.single_mask, None]
    d_ON_vec_single = (
        grid.cell_centers[neighbour_single] - grid.cell_centers[owner_single]
    )
    d_fN_vec_single = (
        grid.cell_centers[neighbour_single]
        - grid.face_centers[psi_f.single_mask]
    )
    d_ON_single = torch.abs(d_ON_vec_single.gather(1, axis_single))
    d_fN_single = torch.abs(d_fN_vec_single.gather(1, axis_single))
    w_single = d_fN_single / d_ON_single
    psi_f.single_data = (
        w_single * field.data[owner_single]
        + (1.0 - w_single) * field.data[neighbour_single]
    )

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
