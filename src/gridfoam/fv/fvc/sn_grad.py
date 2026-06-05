from typing import cast

import torch

from gridfoam.core.field import CellField, FaceField, FieldRole
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.fvc.grad import grad
from gridfoam.fv.fvc.interpolate import interpolate
from gridfoam.fv.mesh_geometry import (
    non_orth_correction_vectors,
    non_orth_delta_coeffs,
)


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
    single_mask = sn_grad_field.single_mask
    owner_single = grid.owner[single_mask]
    neighbour_single = grid.neighbour[single_mask]

    # Internal faces
    d_vec_single = (
        grid.cell_centers[neighbour_single] - grid.cell_centers[owner_single]
    )
    Sf_single = grid.Sf[single_mask]
    mag_Sf_single = cast(
        torch.Tensor,
        torch.linalg.vector_norm(Sf_single, dim=1, keepdim=True),
    )
    delta_coeffs = non_orth_delta_coeffs(d_vec_single, mag_Sf_single, Sf_single)

    psi_N_single = field.data[neighbour_single]
    psi_O_single = field.data[owner_single]
    orthogonal = delta_coeffs * (psi_N_single - psi_O_single)

    corr_vec = non_orth_correction_vectors(
        d_vec_single, delta_coeffs, mag_Sf_single, Sf_single
    )
    grad_f = interpolate(grad(field)).single_data.reshape(
        -1, field.num_components, 3
    )
    correction = torch.sum(corr_vec[:, None, :] * grad_f, dim=2)
    sn_grad_field.single_data = orthogonal + correction

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
