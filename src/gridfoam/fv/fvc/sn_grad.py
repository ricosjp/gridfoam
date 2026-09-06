from gridfoam.core.dimensions import DIM_LENGTH, dim_div
from gridfoam.core.field import (
    CellField,
    FaceField,
    FieldRole,
    get_or_create_facefield,
)
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
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
        field.num_components,
        dimension=dim_div(field.dimension, DIM_LENGTH),
    )
    # Internal faces
    sn_grad_field.single_data = eval_sn_grad(field)

    for batch in iter_boundary_batches(field):
        _, _, ref_g, _ = evaluate_boundary_state(field, batch)
        # Domain boundaries
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            sn_grad_field.domain_bnd_data[batch.face_mask] = ref_g
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                sn_grad_field.immersed_upper[batch.face_mask] = ref_g
                continue
            if batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                sn_grad_field.immersed_lower[batch.face_mask] = ref_g
                continue

    return sn_grad_field
