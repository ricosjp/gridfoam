"""Surface-normal gradient schemes."""

from __future__ import annotations

from collections.abc import Callable

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.shapes import broadcast_entity
from gridfoam.fv.kernels.face_geometry import FaceGeometry, face_geometry
from gridfoam.fv.kernels.face_interpolation import sn_grad_hanging_correction
from gridfoam.fv.kernels.least_squares import least_squares_gradient
from gridfoam.fv.schemes.selection import search_sn_grad_scheme
from gridfoam.meta.enums import SnGradScheme


def uncorrected(
    field: CellField,
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_single *component_shape"]:
    """
    Compact two-point surface-normal gradient on single-sided faces.

    ``(psi_N - psi_O) / |(C_N - C_O) . n|``. Exact on regular faces; on
    hanging-node faces it ignores the tangential offset of the coarse cell
    centre and therefore carries an O(1) error in the normal gradient for
    fields with a tangential gradient component.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Surface-normal gradient with shape ``[F_single, *component_shape]``.
    """
    geo = geometry if geometry is not None else face_geometry(field.grid)
    psi = field.data
    difference = psi[geo.neighbour_s] - psi[geo.owner_s]
    delta_coeffs = broadcast_entity(geo.delta_coeffs_s, difference)
    return delta_coeffs * difference


def corrected(
    field: CellField,
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_single *component_shape"]:
    """
    Skewness-aware surface-normal gradient on single-sided internal faces.

    Each cell value is first reconstructed onto the normal line passing
    through the face centre ``C_f`` using the cell gradient, removing the
    tangential (skew) offset of the cell centre from that line:

    ``psi_O* = psi_O + grad_O & (C_f - C_O)_tangential``,
    ``psi_N* = psi_N + grad_N & (C_f - C_N)_tangential``,

    and the normal gradient is then the compact difference along the face
    normal ``(psi_N* - psi_O*) / |(C_N - C_O) . n|``. The correction is
    non-zero only on hanging-node faces and uses a local least-squares
    gradient of the adjacent cells, so it stays exact for linear fields on
    2:1 octree interfaces. The same correction is used by the ``corrected``
    Laplacian so that ``FvMatrix.flux`` and ``fvc.sn_grad`` agree.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Corrected normal gradient with shape ``[F_single, *component_shape]``.
    """
    geo = geometry if geometry is not None else face_geometry(field.grid)
    base = uncorrected(field, geo)
    if geo.num_hanging == 0:
        return base
    grad_hang = least_squares_gradient(field, geo, hanging_cells_only=True)
    correction = sn_grad_hanging_correction(field, grad_hang, geo)
    result = base.clone()
    result.index_add_(0, geo.hang_idx, correction)
    return result


SnGradSchemeFunc = Callable[
    [CellField, FaceGeometry | None],
    Float[torch.Tensor, " F_single *component_shape"],
]

SN_GRAD_SCHEMES: dict[SnGradScheme, SnGradSchemeFunc] = {
    SnGradScheme.UNCORRECTED: uncorrected,
    SnGradScheme.CORRECTED: corrected,
}


def get_sn_grad_scheme(scheme: SnGradScheme) -> SnGradSchemeFunc:
    """Return the surface-normal-gradient scheme function for the enum."""
    return SN_GRAD_SCHEMES[scheme]


def eval_sn_grad(
    field: CellField,
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_single *component_shape"]:
    """
    Evaluate the configured surface-normal gradient on single-sided faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Surface-normal gradient with shape ``[F_single, *component_shape]``.
    """
    scheme = search_sn_grad_scheme(field.grid.sim_config, field)
    return get_sn_grad_scheme(scheme)(field, geometry)
