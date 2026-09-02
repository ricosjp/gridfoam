"""Cell-centered gradient schemes."""

from collections.abc import Callable

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)
from gridfoam.fv.kernels.face_interpolation import single_internal_mask
from gridfoam.fv.kernels.gauss_gradient import assemble_gauss_gradient
from gridfoam.fv.schemes.interpolate import linear as interpolate_linear
from gridfoam.meta.config import SimulatorConfig
from gridfoam.meta.enums import GradScheme

GradSchemeFunc = Callable[[CellField], Float[torch.Tensor, " C k 3"]]


def _unit_normals(
    vectors: Float[torch.Tensor, " N 3"],
) -> Float[torch.Tensor, " N 3"]:
    mag = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    return vectors / mag


def _add_boundary_least_square_terms(
    field: CellField,
    ata: Float[torch.Tensor, " C 3 3"],
    atb: Float[torch.Tensor, " C 3 k"],
) -> None:
    grid = field.grid
    for batch in iter_boundary_batches(field):
        _, _, _, psi_b = evaluate_boundary_state(field, batch)
        psi_O = field.data[batch.target_cells]
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            face_centers = grid.domain_bnd_face_centers[batch.face_mask]
        elif isinstance(grid, AxisProjectedGrid):
            immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                n_hat = _unit_normals(immersed_Sf[batch.face_mask])
            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                n_hat = _unit_normals(-immersed_Sf[batch.face_mask])
            else:
                continue
            face_centers = (
                grid.cell_centers[batch.target_cells] + n_hat * batch.mag_d
            )
        else:
            continue

        d = face_centers - grid.cell_centers[batch.target_cells]
        dpsi = psi_b - psi_O
        w2 = 1.0 / torch.sum(d * d, dim=1, keepdim=True)
        ata_face = w2[:, :, None] * d[:, :, None] * d[:, None, :]
        atb_face = w2[:, :, None] * d[:, :, None] * dpsi[:, None, :]
        ata.index_add_(0, batch.target_cells, ata_face)
        atb.index_add_(0, batch.target_cells, atb_face)


def linear(field: CellField) -> Float[torch.Tensor, " C k 3"]:
    """
    Green-Gauss gradient from hierarchy-aware linear face values.

    Parameters
    ----------
    field : CellField
        Cell-centered field with ``k`` components.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, k, 3]``.
    """
    psi_f = interpolate_linear(field)
    return assemble_gauss_gradient(field.grid, psi_f)


def leastsquare(field: CellField) -> Float[torch.Tensor, " C k 3"]:
    """
    Weighted least-squares gradient.

    Internal equations use only single-sided faces. Immersed faces are
    never mixed into the owner/neighbour stencil; their upper/lower
    boundary values enter as independent boundary equations.

    Parameters
    ----------
    field : CellField
        Cell-centered field with ``k`` components.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, k, 3]``.
    """
    grid = field.grid
    single_mask = single_internal_mask(grid)
    owner = grid.owner[single_mask]
    neighbour = grid.neighbour[single_mask]
    d = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    dpsi = field.data[neighbour] - field.data[owner]
    w2 = 1.0 / torch.sum(d * d, dim=1, keepdim=True)

    ata_face = w2[:, :, None] * d[:, :, None] * d[:, None, :]
    atb_face = w2[:, :, None] * d[:, :, None] * dpsi[:, None, :]

    ata = torch.zeros(
        (grid.num_cells, 3, 3), dtype=grid.dtype, device=grid.device
    )
    atb = torch.zeros(
        (grid.num_cells, 3, field.num_components),
        dtype=grid.dtype,
        device=grid.device,
    )
    ata.index_add_(0, owner, ata_face)
    ata.index_add_(0, neighbour, ata_face)
    atb.index_add_(0, owner, atb_face)
    atb.index_add_(0, neighbour, atb_face)

    _add_boundary_least_square_terms(field, ata, atb)

    return torch.bmm(torch.linalg.pinv(ata), atb).transpose(1, 2)


GRAD_SCHEMES: dict[GradScheme, GradSchemeFunc] = {
    GradScheme.LINEAR: linear,
    GradScheme.LEASTSQUARE: leastsquare,
}


def get_grad_scheme(scheme: GradScheme) -> GradSchemeFunc:
    """Return the gradient-scheme function for the given enum."""
    return GRAD_SCHEMES[scheme]


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


def eval_grad(field: CellField) -> Float[torch.Tensor, " C k 3"]:
    """
    Evaluate the configured cell-centered gradient as a tensor.

    Schemes that need a gradient must call this helper rather than
    ``fvc.grad``, which wraps the same tensor in a named ``CellField``.

    Parameters
    ----------
    field : CellField
        Target cell-centered field with ``k`` components.

    Returns
    -------
    torch.Tensor
        Cell-centered gradient with shape ``[C, k, 3]``.
    """
    scheme = _search_grad_scheme(field.grid.sim_config, field)
    return get_grad_scheme(scheme)(field)
