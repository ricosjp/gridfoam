import torch

from gridfoam.core.field import CellField, get_or_create_cellfield
from gridfoam.fv.schemes.grad import get_grad_scheme
from gridfoam.meta.config import SimulatorConfig
from gridfoam.meta.enums import GradScheme


def _search_grad_scheme(
    sim_config: SimulatorConfig, field: CellField
) -> GradScheme:
    if sim_config.fvSchemes.gradSchemes is None:
        return GradScheme.LINEAR
    key = f"grad({field.name})"
    grad_scheme = sim_config.fvSchemes.gradSchemes.get(key)
    if grad_scheme is None:
        grad_scheme = sim_config.fvSchemes.gradSchemes.get("default")
    if grad_scheme is None:
        grad_scheme = GradScheme.LINEAR
    return grad_scheme


def grad(field: CellField) -> CellField:
    """
    Compute cell-centered gradient via the configured gradient scheme.

    Supports both scalar and vector fields. Vector fields are decomposed
    into scalar components, each differentiated independently.

    Parameters
    ----------
    field : CellField
        Target cell-centered field.

    Returns
    -------
    CellField
        Computed cell-centered gradient field.
        For scalar input, stores ``num_components=3`` with shape ``[C, 3]``.
        For vector input, stores ``num_components=k * 3`` with shape
        ``[C, k * 3]`` (component ``c`` occupies columns ``3*c:3*c+3``).
    """
    grid = field.grid
    grad_scheme = _search_grad_scheme(grid.sim_config, field)
    scheme_func = get_grad_scheme(grad_scheme)

    # Scalar field case
    if field.num_components == 1:
        # TODO: fix dimension L: -1
        grad_field = get_or_create_cellfield(
            grid, f"grad({field.name})", field.role, 3
        )
        grad_field.data = scheme_func(field)
        return grad_field

    # Vector field case
    grads = []
    for c in range(field.num_components):
        field_c = get_or_create_cellfield(
            grid, f"{field.name}_{c}", field.role, 1
        )
        field_c.data = field.data[:, c : c + 1]

        # Decompose vector BCs per component and apply them
        # to a temporary scalar field.
        field_c_bcs = {k: v.component(c) for k, v in field.bcs.items()}
        field_c.add_boundary_conditions(field_c_bcs)

        grads.append(scheme_func(field_c))

    # Concatenate per-component gradients to [C, k * 3].
    grad_data = torch.cat(grads, dim=1)

    # Gradient of a k-component vector in 3D has k * 3 entries.
    # TODO: fix dimension L: -1
    grad_field = get_or_create_cellfield(
        grid,
        f"grad({field.name})",
        field.role,
        field.num_components * 3,
    )
    grad_field.data = grad_data
    return grad_field
