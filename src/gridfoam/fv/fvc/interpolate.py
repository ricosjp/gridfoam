import torch
from jaxtyping import Float

from gridfoam.core.field import (
    CellField,
    FaceField,
    FieldRole,
    get_or_create_facefield,
)
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.fv.boundary_ops import (
    BoundaryFaceKind,
    evaluate_boundary_state,
    iter_boundary_batches,
)

# Re-entrancy guard: the skew correction needs a provisional cell gradient,
# whose Green-Gauss assembly internally calls ``interpolate``/
# ``linear_internal_face_values`` again. While the guard is active those
# nested calls fall back to the plain (uncorrected) linear interpolation so
# that no unbounded recursion can occur.
_SKEW_REENTRANT = False


def single_internal_mask(grid: IGridBase) -> torch.Tensor:
    """
    Boolean mask selecting single-sided internal faces.

    Immersed (double-sided) internal faces are excluded for
    axis-projected grids.

    Parameters
    ----------
    grid : IGridBase
        Grid providing face connectivity.

    Returns
    -------
    torch.Tensor
        Boolean mask over internal faces with shape ``[F_internal]``.
    """
    single_mask = torch.ones(
        grid.num_internal_faces, dtype=torch.bool, device=grid.device
    )
    if isinstance(grid, AxisProjectedGrid):
        single_mask[grid.ap_is_immersed_faces] = False
    return single_mask


def single_face_linear_weights(
    grid: IGridBase,
    single_mask: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Float[torch.Tensor, " F_single 1"],
]:
    """
    Axis-aligned linear interpolation weights on single-sided faces.

    The interpolated face value is ``psi_f = w * psi_O + (1 - w) * psi_N``.

    Parameters
    ----------
    grid : IGridBase
        Grid providing cell centers and face centers.
    single_mask : torch.Tensor
        Boolean mask over internal faces selecting single-sided faces.

    Returns
    -------
    tuple
        ``(owner, neighbour, w)`` indexed by the single-sided face subset.
    """
    owner = grid.owner[single_mask]
    neighbour = grid.neighbour[single_mask]
    axis_idx = grid.axis[single_mask, None]
    d_ON_vec = grid.cell_centers[neighbour] - grid.cell_centers[owner]
    d_fN_vec = grid.cell_centers[neighbour] - grid.face_centers[single_mask]
    d_ON = torch.abs(d_ON_vec.gather(1, axis_idx))
    d_fN = torch.abs(d_fN_vec.gather(1, axis_idx))
    w = d_fN / d_ON
    return owner, neighbour, w


def single_face_skew_correction(
    field: CellField,
    grad_data: torch.Tensor,
    single_mask: torch.Tensor,
    owner: torch.Tensor,
    neighbour: torch.Tensor,
    w: Float[torch.Tensor, " F_single 1"],
) -> Float[torch.Tensor, " F_single k"]:
    """
    Face-centre offset (skewness) correction for linear interpolation.

    The axis-aligned linear estimate implicitly evaluates the field at the
    interpolated centroid ``C_w = w * C_O + (1 - w) * C_N`` located on the
    owner-neighbour line, whereas the true face centre ``C_f`` is offset from
    that line on hanging-node (2:1) octree interfaces. The offset is corrected
    with the interpolated face gradient:

    ``correction = grad_f & (C_f - C_w)``,

    where ``grad_f = w * grad_O + (1 - w) * grad_N``. On uniform meshes
    ``C_f = C_w`` and the correction vanishes identically.

    Parameters
    ----------
    field : CellField
        Cell-centered field being interpolated.
    grad_data : torch.Tensor
        Provisional cell-centered gradient with shape ``[C, k * 3]``.
    single_mask : torch.Tensor
        Boolean mask selecting single-sided internal faces.
    owner, neighbour : torch.Tensor
        Owner and neighbour cell indices on the single-sided faces.
    w : torch.Tensor
        Axis-aligned linear interpolation weights with shape ``[F_single, 1]``.

    Returns
    -------
    torch.Tensor
        Skewness correction with shape ``[F_single, k]``.
    """
    grid = field.grid
    k = field.num_components
    grad_o = grad_data[owner].reshape(-1, k, 3)
    grad_n = grad_data[neighbour].reshape(-1, k, 3)
    grad_face = w[:, None] * grad_o + (1.0 - w)[:, None] * grad_n

    centroid = (
        w * grid.cell_centers[owner] + (1.0 - w) * grid.cell_centers[neighbour]
    )
    face_offset = grid.face_centers[single_mask] - centroid
    return torch.sum(grad_face * face_offset[:, None, :], dim=2)


def linear_internal_face_values(
    field: CellField,
    grad_data: torch.Tensor | None = None,
) -> Float[torch.Tensor, " F_single k"]:
    """
    Linearly interpolate cell values to single-sided internal faces.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    grad_data : torch.Tensor, optional
        Provisional cell-centered gradient with shape ``[C, k * 3]``. When
        provided, a face-centre offset (skewness) correction is added so that
        the interpolation stays second-order accurate on hanging-node (2:1)
        octree interfaces. When ``None`` the plain axis-aligned linear
        interpolation is returned (default).

    Returns
    -------
    torch.Tensor
        Interpolated face values with shape ``[F_single, k]``.
    """
    grid = field.grid
    single_mask = single_internal_mask(grid)
    owner, neighbour, w = single_face_linear_weights(grid, single_mask)
    psi_linear = w * field.data[owner] + (1.0 - w) * field.data[neighbour]
    if grad_data is None:
        return psi_linear
    return psi_linear + single_face_skew_correction(
        field, grad_data, single_mask, owner, neighbour, w
    )


def _provisional_grad_data(
    field: CellField,
) -> torch.Tensor:
    """
    Compute a provisional cell gradient for the skewness correction.

    The re-entrancy guard is raised for the duration of the call so that the
    gradient's internal Green-Gauss assembly uses the plain (uncorrected)
    linear interpolation and cannot recurse back into the skew correction.

    Parameters
    ----------
    field : CellField
        Cell-centered field being interpolated.

    Returns
    -------
    torch.Tensor
        Provisional cell-centered gradient with shape ``[C, k * 3]``.
    """
    global _SKEW_REENTRANT
    from gridfoam.fv.fvc.grad import grad

    _SKEW_REENTRANT = True
    try:
        return grad(field).data
    finally:
        _SKEW_REENTRANT = False


def _use_skew_correction(field: CellField) -> bool:
    """
    Decide whether skewness correction should be applied for ``field``.

    The correction is enabled for primary cell fields (scalars and vectors),
    but skipped for gradient/tensor fields (identified by a ``grad(`` name
    prefix or more than three components) to avoid computing a gradient of a
    gradient, and while the re-entrancy guard is active.

    Parameters
    ----------
    field : CellField
        Cell-centered field being interpolated.

    Returns
    -------
    bool
        Whether to apply the face-centre offset correction.
    """
    if _SKEW_REENTRANT:
        return False
    if field.num_components > 3:
        return False
    if field.name.startswith("grad("):
        return False
    return True


def interpolate(
    field: CellField, *, skew_correction: bool | None = None
) -> FaceField:
    """
    Linearly interpolate cell-centered values to face centers.

    When APIBM (Axis Projected Immersed Boundary Method) is enabled,
    boundary values are updated with immersed-boundary-aware states.

    On hanging-node (2:1) octree interfaces the face centre does not lie on
    the owner-neighbour line, so a face-centre offset (skewness) correction is
    applied to keep the interpolation second-order accurate. The correction
    vanishes identically on uniform meshes.

    Parameters
    ----------
    field : CellField
        Cell-centered field.
    skew_correction : bool, optional
        Whether to apply the face-centre offset (skewness) correction on
        internal faces. When ``None`` (default) it is enabled automatically for
        primary scalar/vector fields (see :func:`_use_skew_correction`).

    Returns
    -------
    FaceField
        Interpolated face-centered field.
    """
    grid = field.grid

    if skew_correction is None:
        skew_correction = _use_skew_correction(field)
    grad_data = _provisional_grad_data(field) if skew_correction else None

    psi_f = get_or_create_facefield(
        grid, f"{field.name}_f", FieldRole.LOCAL, field.num_components
    )
    # Internal faces
    psi_f.single_data = linear_internal_face_values(field, grad_data=grad_data)

    for batch in iter_boundary_batches(field):
        _, _, _, psi_b = evaluate_boundary_state(field, batch)
        # Domain boundaries
        if batch.face_kind == BoundaryFaceKind.DOMAIN:
            psi_f.domain_bnd_data[batch.face_mask] = psi_b
            continue

        # Immersed boundaries
        if isinstance(grid, AxisProjectedGrid):
            if batch.face_kind == BoundaryFaceKind.IMMERSED_UPPER:
                psi_f.immersed_upper[batch.face_mask] = psi_b
                continue
            elif batch.face_kind == BoundaryFaceKind.IMMERSED_LOWER:
                psi_f.immersed_lower[batch.face_mask] = psi_b
                continue

    return psi_f
