"""Low-level kernels for interpolation onto internal faces."""

from __future__ import annotations

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField
from gridfoam.core.shapes import broadcast_entity, require_shape
from gridfoam.fv.kernels.face_geometry import FaceGeometry, face_geometry


def linear_internal_face_values(
    field: CellField,
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_single *component_shape"]:
    """
    Uncorrected axis-aligned linear values on single-sided internal faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Face values with shape ``[F_single, *component_shape]``.
    """
    geo = geometry if geometry is not None else face_geometry(field.grid)
    psi = field.data
    psi_owner = psi[geo.owner_s]
    psi_neighbour = psi[geo.neighbour_s]
    w = broadcast_entity(geo.w_s, psi_owner)
    return w * psi_owner + (1.0 - w) * psi_neighbour


def correct_internal_values(
    field: CellField,
    base_values: Float[torch.Tensor, " F_single *component_shape"],
    grad_data: Float[torch.Tensor, " C *component_shape 3"],
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_single *component_shape"]:
    """
    Add face-centre offset correction to base linear face values.

    ``psi_f = base + grad_f & (C_f - C_w)``,

    where ``C_w`` is the weighted cell-centre interpolation point and
    ``grad_f`` is the same weighted interpolation of the cell gradient.
    The offset vanishes on regular faces, so the correction is evaluated
    only on hanging-node faces; ``grad_data`` needs to be valid only on the
    cells adjacent to those faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    base_values : torch.Tensor
        Uncorrected face values with shape ``[F_single, *component_shape]``.
    grad_data : torch.Tensor
        Cell-centered gradient with shape ``[C, *component_shape, 3]``.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Corrected face values with shape ``[F_single, *component_shape]``.
    """
    grid = field.grid
    geo = geometry if geometry is not None else face_geometry(grid)
    require_shape(
        grad_data, (grid.num_cells, *field.component_shape, 3), "gradient"
    )
    require_shape(
        base_values, (geo.num_single, *field.component_shape), "face values"
    )
    if geo.num_hanging == 0:
        return base_values

    hang = geo.hang_idx
    owner = geo.owner_s[hang]
    neighbour = geo.neighbour_s[hang]
    grad_owner = grad_data[owner]
    grad_neighbour = grad_data[neighbour]
    w = broadcast_entity(geo.w_s[hang], grad_owner)
    grad_face = w * grad_owner + (1.0 - w) * grad_neighbour
    # Contract the differentiation axis with the face-centre displacement.
    correction = torch.einsum("n...j,nj->n...", grad_face, geo.hang_offset)

    corrected = base_values.clone()
    corrected.index_add_(0, hang, correction)
    return corrected


def sn_grad_hanging_correction(
    field: CellField,
    grad_data: Float[torch.Tensor, " C *component_shape 3"],
    geometry: FaceGeometry | None = None,
) -> Float[torch.Tensor, " F_hang *component_shape"]:
    """
    Skewness correction of the surface-normal gradient on hanging faces.

    Each cell value is reconstructed onto the face-normal line through the
    face centre using the cell gradient, removing the tangential offset of
    the cell centre from that line. The correction to the compact
    difference ``(psi_N - psi_O) / |d . n|`` is

    ``(grad_N & t_N - grad_O & t_O) / |d . n|``,

    with ``t_O = (C_f - C_O)`` and ``t_N = (C_f - C_N)`` projected onto the
    face plane. It is exact for linear fields whenever the supplied
    gradient is, and is zero on regular faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    grad_data : torch.Tensor
        Cell gradient ``[C, *component_shape, 3]``, valid on cells
        adjacent to hanging faces.
    geometry : FaceGeometry or None
        Cached face geometry. Looked up from the grid when ``None``.

    Returns
    -------
    torch.Tensor
        Hanging-face normal-gradient correction, ``[F_hang, *component_shape]``.
    """
    geo = geometry if geometry is not None else face_geometry(field.grid)
    hang = geo.hang_idx
    owner = geo.owner_s[hang]
    neighbour = geo.neighbour_s[hang]
    correction_neighbour = torch.einsum(
        "n...j,nj->n...", grad_data[neighbour], geo.hang_move_N
    )
    correction_owner = torch.einsum(
        "n...j,nj->n...", grad_data[owner], geo.hang_move_O
    )
    difference = correction_neighbour - correction_owner
    delta_coeffs = broadcast_entity(geo.delta_coeffs_s[hang], difference)
    return delta_coeffs * difference
