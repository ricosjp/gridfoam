import torch

from gridfoam.core.field import (
    CellField,
    FaceField,
    FieldRole,
    get_or_create_cellfield,
)
from gridfoam.core.grid.axis_projected import AxisProjectedGrid


def div(phi: FaceField) -> CellField:
    """
    Compute cell-centered divergence from a face field.

    Parameters
    ----------
    phi : FaceField
        Face-centered input field.

    Returns
    -------
    CellField
        Cell-centered divergence field.
    """
    grid = phi.grid
    # Volume-integrated form (no division by cell volume), matching the
    # ``fvm`` operators so that both sides of an equation are consistent.
    div_phi = get_or_create_cellfield(
        grid,
        f"div({phi.name})",
        FieldRole.LOCAL,
        1,
        dimension=phi.dimension,
    )
    data = torch.zeros(
        (grid.num_cells, 1), dtype=grid.dtype, device=grid.device
    )

    # Internal faces
    single_data = phi.single_data.clone()
    data.index_add_(0, grid.owner[phi.single_mask], single_data)
    data.index_add_(0, grid.neighbour[phi.single_mask], -single_data)

    # Domain boundaries
    data.index_add_(0, grid.domain_bnd_owner, phi.domain_bnd_data.clone())

    # Immersed boundaries
    if isinstance(grid, AxisProjectedGrid):
        data.index_add_(
            0, grid.owner[grid.ap_is_immersed_faces], phi.immersed_upper.clone()
        )
        data.index_add_(
            0,
            grid.neighbour[grid.ap_is_immersed_faces],
            -phi.immersed_lower.clone(),
        )
    div_phi.data = data
    return div_phi
