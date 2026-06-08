from gridfoam.core.field import CellField, FaceField, FieldRole
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.fvc.reconstruction import corrected_internal_sn_grad_values


def sn_grad(field: CellField) -> FaceField:
    """
    Compute the surface-normal gradient on faces.

    Uses the OpenFOAM corrected scheme:

    snGrad(psi) = nonOrthDeltaCoeffs * (psi_N - psi_O)
                  + nonOrthCorrectionVectors & grad(psi)_f.

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

    sn_grad_field = grid.get_field(f"snGrad({field.name})")
    if sn_grad_field is None:
        sn_grad_field = FaceField(
            grid,
            name=f"snGrad({field.name})",
            role=FieldRole.LOCAL,
            num_components=field.num_components,
            dimension=field.dimension,  # TODO: fix L: -1
            export=False,
        )
    assert isinstance(sn_grad_field, FaceField)
    # Internal faces
    sn_grad_field.single_data = corrected_internal_sn_grad_values(field)

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
