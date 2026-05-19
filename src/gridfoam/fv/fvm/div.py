import logging

import torch

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.schemes.div import get_div_scheme
from gridfoam.meta.config import SimulatorConfig
from gridfoam.meta.enums import DivScheme

logger = logging.getLogger(__name__)


def _search_div_scheme(
    sim_config: SimulatorConfig, phi: FaceField, field: CellField
) -> DivScheme:
    if sim_config.fvSchemes.divSchemes is None:
        return DivScheme.UPWIND
    key = f"div({phi.name}, {field.name})"
    div_scheme = sim_config.fvSchemes.divSchemes.get(key)
    if div_scheme is None:
        div_scheme = sim_config.fvSchemes.divSchemes.get("default")
    if div_scheme is None:
        logger.warning(f"Div scheme for {key} not found. Using UPWIND scheme.")
        div_scheme = DivScheme.UPWIND
    return div_scheme


def div(phi: FaceField, field: CellField) -> FvMatrix:
    """
    Build the convection (divergence) matrix term.

    Represents div(U * psi).

    Parameters
    ----------
    phi : FaceField
        Face mass/volumetric flux field.
    field : CellField
        Advected cell-centered field.

    Returns
    -------
    FvMatrix
        Coefficient matrix assembled from convection term.
    """
    mat = FvMatrix(field)
    grid = field.grid
    div_scheme = _search_div_scheme(grid.sim_config, phi, field)

    # Internal faces
    scheme_func = get_div_scheme(div_scheme)
    upper, lower, diag_O, diag_N, source_face = scheme_func(phi, field)

    single_mask = phi.single_mask
    mat.upper[single_mask] = upper
    mat.lower[single_mask] = lower
    mat.diag.index_add_(0, grid.owner[single_mask], diag_O)
    mat.diag.index_add_(0, grid.neighbour[single_mask], diag_N)
    mat.source.index_add_(0, grid.owner[single_mask], source_face)
    mat.source.index_add_(0, grid.neighbour[single_mask], -source_face)

    for batch in iter_boundary_batches(field):
        f, ref_v, ref_g, _ = evaluate_boundary_state(field, batch)
        # Domain boundaries
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            flux = phi.domain_bnd_data[batch.face_mask]
            F_out = torch.clamp(flux, min=0.0)
            F_in = torch.clamp(flux, max=0.0)
            diag = F_out + F_in * (1.0 - f)
            src = F_in * (f * ref_v + (1.0 - f) * ref_g * batch.mag_d)
            mat.diag.index_add_(0, batch.target_cells, diag)
            mat.source.index_add_(0, batch.target_cells, -src)
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                flux = phi.immersed_upper[batch.face_mask]
                F_out = torch.clamp(flux, min=0.0)
                F_in = torch.clamp(flux, max=0.0)

                wb = grid.ap_owner_weights[batch.face_mask, 0:1]
                w = grid.ap_owner_weights[batch.face_mask, 1:2]

                diag = F_out + F_in * (w + wb * (1.0 - f))
                src = F_in * wb * (f * ref_v + (1.0 - f) * ref_g * batch.mag_d)
                mat.diag.index_add_(0, batch.target_cells, diag)
                mat.source.index_add_(0, batch.target_cells, -src)
                continue
            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                flux = phi.immersed_lower[batch.face_mask]
                F_out = torch.clamp(flux, min=0.0)
                F_in = torch.clamp(flux, max=0.0)

                wb = grid.ap_neighbour_weights[batch.face_mask, 0:1]
                w = grid.ap_neighbour_weights[batch.face_mask, 1:2]

                diag = F_out + F_in * (w + wb * (1.0 - f))
                src = F_in * wb * (f * ref_v + (1.0 - f) * ref_g * batch.mag_d)
                mat.diag.index_add_(0, batch.target_cells, diag)
                mat.source.index_add_(0, batch.target_cells, -src)
                continue

    return mat
