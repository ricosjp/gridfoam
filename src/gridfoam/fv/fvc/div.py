from gridfoam.core.field import CellField, FaceField, FieldRole
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
    div_phi = grid.get_field(f"div({phi.name})")
    if div_phi is None:
        div_phi = CellField(
            grid,
            f"div({phi.name})",
            role=FieldRole.LOCAL,
            num_components=1,
            dimension=phi.dimension,
            export=phi.export,
        )
    assert isinstance(div_phi, CellField)

    # Clear previous values (index_add_ is cumulative)
    div_phi.data.zero_()

    # Internal faces
    div_phi.data.index_add_(0, grid.owner[phi.single_mask], phi.single_data)
    div_phi.data.index_add_(
        0, grid.neighbour[phi.single_mask], -phi.single_data
    )

    # Domain boundaries
    div_phi.data.index_add_(0, grid.domain_bnd_owner, phi.domain_bnd_data)

    # Immersed boundaries
    if isinstance(grid, AxisProjectedGrid):
        div_phi.data.index_add_(
            0, grid.owner[grid.ap_is_immersed_faces], phi.immersed_upper
        )
        div_phi.data.index_add_(
            0, grid.neighbour[grid.ap_is_immersed_faces], -phi.immersed_lower
        )
    return div_phi
