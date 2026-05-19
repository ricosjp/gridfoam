import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.fvc.interpolate import interpolate


def grad(field: CellField) -> CellField:
    """
    Compute cell-centered gradient via the Gauss theorem.

    Supports both scalar and vector fields:
    grad(psi) ~= (1 / V) * sum(Sf * psi_f).

    Parameters
    ----------
    field : CellField
        Target cell-centered field.

    Returns
    -------
    CellField
        Computed cell-centered gradient field.
        For scalar input, stores ``num_components=3`` with shape ``[C, 3]``.
        For vector input, stores shape ``[C, k, 3]``.
    """
    grid = field.grid
    # Scalar field case
    if field.num_components == 1:
        grad_data = _grad_scalar(field)
        grad_field = grid.get_field(f"grad({field.name})")
        if grad_field is None:
            grad_field = CellField(
                grid,
                name=f"grad({field.name})",
                role=field.role,
                num_components=3,
                dimension=field.dimension,  # TODO: fix L: -1
                export=field.export,
            )
        assert isinstance(grad_field, CellField)
        grad_field.data = grad_data
        return grad_field

    # Vector field case
    grads = []
    for c in range(field.num_components):
        field_c = grid.get_field(f"{field.name}_{c}")
        if field_c is None:
            field_c = CellField(
                grid,
                name=f"{field.name}_{c}",
                role=field.role,
                num_components=1,
                dimension=field.dimension,
                export=False,
            )
        assert isinstance(field_c, CellField)
        field_c.data = field.data[:, c : c + 1]

        # Decompose vector BCs per component and apply them
        # to a temporary scalar field.
        field_c_bcs = {k: v.component(c) for k, v in field.bcs.items()}
        field_c.add_boundary_conditions(field_c_bcs)

        grad_c_data = _grad_scalar(field_c)
        grads.append(grad_c_data)

    # Stack per-component gradients to build a [C, k, 3] tensor.
    grad_data = torch.stack(grads, dim=1)

    # Gradient of a k-component vector in 3D has k * 3 entries.
    grad_field = grid.get_field(f"grad({field.name})")
    if grad_field is None:
        grad_field = CellField(
            grid,
            name=f"grad({field.name})",
            role=field.role,
            num_components=field.num_components * 3,
            dimension=field.dimension,  # TODO: fix L: -1
            export=field.export,
        )
    assert isinstance(grad_field, CellField)
    grad_field.data = grad_data
    return grad_field


def _grad_scalar(field: CellField) -> Float[torch.Tensor, " C 3"]:
    """
    Internal Gauss-theorem gradient routine for scalar fields.

    Parameters
    ----------
    field : CellField
        Target cell-centered scalar field.

    Returns
    -------
    torch.Tensor
            Computed cell-centered gradient tensor with shape ``[C, 3]``.
    """
    grid = field.grid
    assert field.num_components == 1
    psi_f = interpolate(field)

    grad_data = torch.zeros(
        (grid.num_cells, 3), dtype=grid.dtype, device=grid.device
    )

    # Contributions from internal faces
    single_Sf = grid.Sf[psi_f.single_mask]
    flux = single_Sf * psi_f.single_data
    grad_data.index_add_(0, grid.owner[psi_f.single_mask], flux)
    grad_data.index_add_(0, grid.neighbour[psi_f.single_mask], -flux)

    # Contributions from domain boundaries
    flux_domain_bnd = grid.domain_bnd_Sf * psi_f.domain_bnd_data
    grad_data.index_add_(0, grid.domain_bnd_owner, flux_domain_bnd)

    # Contributions from immersed boundaries
    if isinstance(grid, AxisProjectedGrid):
        immersed_owner = grid.owner[grid.ap_is_immersed_faces]
        immersed_neighbour = grid.neighbour[grid.ap_is_immersed_faces]
        immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
        flux_immersed_upper = immersed_Sf * psi_f.immersed_upper
        flux_immersed_lower = immersed_Sf * psi_f.immersed_lower
        grad_data.index_add_(0, immersed_owner, flux_immersed_upper)
        grad_data.index_add_(0, immersed_neighbour, -flux_immersed_lower)

    # Divide by control-volume size
    grad_data = grad_data / grid.cell_volumes
    return grad_data
