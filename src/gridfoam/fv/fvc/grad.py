"""Public cell-centered gradient entry point."""

from gridfoam.core.dimensions import DIM_LENGTH, dim_div
from gridfoam.core.field import CellField, get_or_create_cellfield
from gridfoam.fv.schemes.grad import eval_grad


def grad(field: CellField) -> CellField:
    """
    Compute cell-centered gradient via the configured gradient scheme.

    Parameters
    ----------
    field : CellField
        Target cell-centered field.

    Returns
    -------
    CellField
        Gradient field named ``grad({field.name})`` with
        ``component_shape = field.component_shape + (3,)``. The last axis
        is the differentiation direction.
    """
    grid = field.grid
    grad_tensor = eval_grad(field)
    grad_field = get_or_create_cellfield(
        grid,
        f"grad({field.name})",
        field.role,
        field.component_shape + (3,),
        dimension=dim_div(field.dimension, DIM_LENGTH),
    )
    grad_field.data = grad_tensor
    return grad_field
