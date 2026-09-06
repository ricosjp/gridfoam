import torch

from gridfoam.core.dimensions import DIM_VOLUME, dim_div
from gridfoam.core.field import (
    CellField,
    FaceField,
    FieldRole,
    get_or_create_cellfield,
)
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.kernels.face_geometry import face_geometry


def div(phi: FaceField) -> CellField:
    """
    Compute cell-centered divergence from a face field.

    Implements OpenFOAM ``fvc::div``, i.e. the face sum normalized by the
    cell volume. Callers adding the result to an ``FvMatrix`` source must
    multiply by the cell volume, as OpenFOAM ``fvMatrix::operator==`` does.

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
    div_phi = get_or_create_cellfield(
        grid,
        f"div({phi.name})",
        FieldRole.LOCAL,
        1,
        dimension=dim_div(phi.dimension, DIM_VOLUME),
    )
    data = torch.zeros(
        (grid.num_cells, 1), dtype=grid.dtype, device=grid.device
    )

    # Internal faces
    geo = face_geometry(grid)
    data.index_add_(0, geo.owner_s, phi.single_data)
    data.index_add_(0, geo.neighbour_s, -phi.single_data)

    # Domain boundaries
    data.index_add_(0, grid.domain_bnd_owner, phi.domain_bnd_data)

    # Immersed boundaries: both blocks are stored outward from their
    # adjacent cell (upper along +Sf for the owner, lower along -Sf for the
    # neighbour), so both enter with a positive sign.
    if isinstance(grid, AxisProjectedGrid) and grid.num_immersed_faces > 0:
        immersed = grid.ap_is_immersed_faces
        data.index_add_(0, grid.owner[immersed], phi.immersed_upper)
        data.index_add_(0, grid.neighbour[immersed], phi.immersed_lower)
    div_phi.data = data / grid.cell_volumes
    return div_phi
