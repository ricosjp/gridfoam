from gridfoam.core.dimensions import DIM_LENGTH, dim_div
from gridfoam.core.field import (
    CellField,
    FaceField,
    FieldRole,
    get_or_create_facefield,
)
from gridfoam.fv.boundary_ops import (
    boundary_block,
    boundary_normal_gradient,
    iter_boundary_states,
)
from gridfoam.fv.schemes.sn_grad import eval_sn_grad


def sn_grad(field: CellField) -> FaceField:
    """
    Compute the surface-normal gradient on faces.

    Internal faces use the scheme configured in ``snGradSchemes``
    (``corrected`` by default, which adds the hanging-node skewness
    correction). Boundary faces carry the boundary-condition normal
    gradient.

    Parameters
    ----------
    field : CellField
        Cell-centered scalar field.

    Returns
    -------
    FaceField
        Face-centered scalar field.
    """
    grid = field.grid

    sn_grad_field = get_or_create_facefield(
        grid,
        f"snGrad({field.name})",
        FieldRole.LOCAL,
        field.component_shape,
        dimension=dim_div(field.dimension, DIM_LENGTH),
    )
    # Internal faces
    sn_grad_field.single_data = eval_sn_grad(field)

    for state in iter_boundary_states(field):
        batch = state.batch
        boundary_block(sn_grad_field, batch.face_kind)[batch.face_mask] = (
            boundary_normal_gradient(field, state)
        )

    return sn_grad_field
