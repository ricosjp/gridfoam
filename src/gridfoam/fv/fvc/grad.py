"""Public cell-centered gradient entry point."""

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

    Parameters
    ----------
    field : CellField
        Target cell-centered field with ``k`` components.

    Returns
    -------
    CellField
        Gradient field named ``grad({field.name})`` with
        ``num_components = k * 3``. Component ``c`` occupies columns
        ``3 * c : 3 * c + 3``.
    """
    grid = field.grid
    grad_scheme = _search_grad_scheme(grid.sim_config, field)
    scheme_func = get_grad_scheme(grad_scheme)

    grad_tensor = scheme_func(field)
    grad_field = get_or_create_cellfield(
        grid,
        f"grad({field.name})",
        field.role,
        field.num_components * 3,
    )
    grad_field.data = grad_tensor.reshape(
        grid.num_cells, field.num_components * 3
    )
    return grad_field
