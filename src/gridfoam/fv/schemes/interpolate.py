"""Face-interpolation schemes."""

from __future__ import annotations

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
from gridfoam.fv.kernels.face_interpolation import (
    correct_internal_values,
    linear_internal_face_values,
)
from gridfoam.fv.kernels.gauss_gradient import assemble_gauss_gradient


def _fill_boundary_face_values(field: CellField, psi_f: FaceField) -> None:
    """Write domain and immersed boundary values into ``psi_f``."""
    grid = field.grid
    for batch in iter_boundary_batches(field):
        _, _, _, psi_b = evaluate_boundary_state(field, batch)
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            psi_f.domain_bnd_data[batch.face_mask] = psi_b
            continue
        if isinstance(grid, AxisProjectedGrid):
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                psi_f.immersed_upper[batch.face_mask] = psi_b
            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                psi_f.immersed_lower[batch.face_mask] = psi_b


def linear(field: CellField) -> FaceField:
    """
    Interpolate a cell field to faces with hierarchy-aware linear weights.

    Gridfoam ``linear`` includes the face-centre offset correction used for
    hanging-node interfaces. A provisional Green-Gauss gradient is formed
    from uncorrected linear face values and then used only as an explicit
    correction; the public ``fvc.grad`` entry point is never called.

    Parameters
    ----------
    field : CellField
        Cell-centered field.

    Returns
    -------
    FaceField
        Face-centered field named ``{field.name}_f``.
    """
    grid = field.grid
    psi_f = get_or_create_facefield(
        grid,
        f"{field.name}_f",
        FieldRole.LOCAL,
        field.num_components,
        dimension=field.dimension,
    )

    base_values = linear_internal_face_values(field)
    psi_f.single_data = base_values
    _fill_boundary_face_values(field, psi_f)

    provisional = assemble_gauss_gradient(grid, psi_f)
    psi_f.single_data = correct_internal_values(field, base_values, provisional)
    return psi_f
