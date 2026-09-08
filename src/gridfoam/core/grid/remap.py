"""Nearest-neighbour transfer of volume fields after a topology change."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gridfoam.core.grid.base import GridBase
from gridfoam.core.shapes import broadcast_entity


@dataclass(frozen=True)
class VolumeFieldSnapshot:
    """Old-mesh samples of ``U``, ``p``, and ``phi`` used after remesh."""

    cell_centers: torch.Tensor
    u: torch.Tensor
    p: torch.Tensor
    face_centers: torch.Tensor
    sf: torch.Tensor
    single_mask: torch.Tensor
    phi_single: torch.Tensor
    domain_centers: torch.Tensor
    domain_sf: torch.Tensor
    phi_domain: torch.Tensor


def capture_volume_fields(grid: GridBase) -> VolumeFieldSnapshot:
    """
    Copy ``U``, ``p``, and ``phi`` together with the current mesh geometry.

    Parameters
    ----------
    grid : GridBase
        Grid whose registered fields are snapshotted.

    Returns
    -------
    VolumeFieldSnapshot
        Geometry and field tensors cloned from ``grid``.

    Raises
    ------
    ValueError
        If ``U``, ``p``, or ``phi`` is not registered.
    """
    u_field = grid.get_cellfield("U")
    p_field = grid.get_cellfield("p")
    phi = grid.get_facefield("phi")
    if u_field is None or p_field is None or phi is None:
        raise ValueError("U, p, and phi must be registered before remesh")
    return VolumeFieldSnapshot(
        cell_centers=grid.cell_centers.clone(),
        u=u_field.data.clone(),
        p=p_field.data.clone(),
        face_centers=grid.face_centers.clone(),
        sf=grid.Sf.clone(),
        single_mask=phi.single_mask.clone(),
        phi_single=phi.single_data.clone(),
        domain_centers=grid.domain_bnd_face_centers.clone(),
        domain_sf=grid.domain_bnd_Sf.clone(),
        phi_domain=phi.domain_bnd_data.clone(),
    )


def map_volume_fields(
    grid: GridBase,
    snapshot: VolumeFieldSnapshot,
) -> None:
    """
    Nearest-neighbour map of ``U``, ``p``, and ``phi`` onto a remeshed grid.

    Cell values are copied from the closest old cell centre. Face fluxes are
    copied from the closest old face and rescaled by
    ``(Sf_new · Sf_old) / |Sf_old|^2`` so orientation and area changes are
    accounted for.

    Parameters
    ----------
    grid : GridBase
        Destination grid after ``remesh``.
    snapshot : VolumeFieldSnapshot
        Source geometry and fields captured before ``remesh``.
    """
    u_field = grid.get_cellfield("U")
    p_field = grid.get_cellfield("p")
    phi = grid.get_facefield("phi")
    if u_field is None or p_field is None or phi is None:
        raise ValueError("U, p, and phi must be registered after remesh")

    cell_idx = _nearest_indices(snapshot.cell_centers, grid.cell_centers)
    u_field.data = snapshot.u[cell_idx]
    p_field.data = snapshot.p[cell_idx]
    u_field.update_history(reset=True)
    p_field.update_history(reset=True)

    src_faces = snapshot.face_centers[snapshot.single_mask]
    src_sf = snapshot.sf[snapshot.single_mask]
    dst_mask = phi.single_mask
    phi.single_data = _mapped_flux(
        src_faces,
        src_sf,
        snapshot.phi_single,
        grid.face_centers[dst_mask],
        grid.Sf[dst_mask],
    )
    phi.domain_bnd_data = _mapped_flux(
        snapshot.domain_centers,
        snapshot.domain_sf,
        snapshot.phi_domain,
        grid.domain_bnd_face_centers,
        grid.domain_bnd_Sf,
    )
    phi.update_history(reset=True)


def _nearest_indices(
    src_xyz: torch.Tensor,
    dst_xyz: torch.Tensor,
) -> torch.Tensor:
    """Return the nearest source index for each destination point."""
    if src_xyz.shape[0] == 0:
        raise ValueError("source point set is empty")
    return torch.cdist(dst_xyz, src_xyz).argmin(dim=1)


def _mapped_flux(
    src_xyz: torch.Tensor,
    src_sf: torch.Tensor,
    src_phi: torch.Tensor,
    dst_xyz: torch.Tensor,
    dst_sf: torch.Tensor,
) -> torch.Tensor:
    """Map volumetric flux onto new faces with area-vector rescaling."""
    n_dst = dst_xyz.shape[0]
    component_shape = src_phi.shape[1:]
    if n_dst == 0:
        return src_phi.new_zeros((0, *component_shape))
    if src_xyz.shape[0] == 0:
        return src_phi.new_zeros((n_dst, *component_shape))
    idx = _nearest_indices(src_xyz, dst_xyz)
    src_sf_m = src_sf[idx]
    denom = torch.sum(src_sf_m * src_sf_m, dim=1).clamp_min(1.0e-30)
    factor = torch.sum(dst_sf * src_sf_m, dim=1) / denom
    mapped_values = src_phi[idx]
    factor_view = broadcast_entity(factor, mapped_values)
    return factor_view * mapped_values
