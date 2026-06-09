from __future__ import annotations

import torch
from jaxtyping import Float, Int

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.post.forces.sampling import SurfaceSample, iter_patch_side_samples


def reset_surface_force_data(grid: AxisProjectedGrid) -> None:
    """
    Initialize per-surface-face force fields for the current evaluation.

    Parameters
    ----------
    grid : AxisProjectedGrid
        Grid owning the surface mesh.
    """
    surface_mesh = grid.surface_mesh
    surface_mesh.cell_data["pressure_force"] = torch.zeros(
        (surface_mesh.n_cells, 3),
        dtype=grid.dtype,
        device=grid.device,
    )
    surface_mesh.cell_data["viscous_force"] = torch.zeros(
        (surface_mesh.n_cells, 3),
        dtype=grid.dtype,
        device=grid.device,
    )


def accumulate_sample_forces_on_surface_mesh(
    grid: AxisProjectedGrid,
    anchor_id: Int[torch.Tensor, " F_sample"],
    pressure_force: Float[torch.Tensor, " F_sample 3"],
    viscous_force: Float[torch.Tensor, " F_sample 3"],
) -> None:
    """
    Accumulate AP-sample forces onto anchored surface faces.

    Parameters
    ----------
    grid : AxisProjectedGrid
        Grid owning the surface mesh.
    anchor_id : torch.Tensor
        Surface-mesh face ids associated with the AP samples.
    pressure_force : torch.Tensor
        Pressure-force contribution on AP samples.
    viscous_force : torch.Tensor
        Viscous-force contribution on AP samples.
    """
    surface_mesh = grid.surface_mesh

    pressure_data = surface_mesh.cell_data.get("pressure_force")
    if pressure_data is None:
        pressure_data = torch.zeros(
            (surface_mesh.n_cells, 3),
            dtype=grid.dtype,
            device=grid.device,
        )
        surface_mesh.cell_data["pressure_force"] = pressure_data

    viscous_data = surface_mesh.cell_data.get("viscous_force")
    if viscous_data is None:
        viscous_data = torch.zeros(
            (surface_mesh.n_cells, 3),
            dtype=grid.dtype,
            device=grid.device,
        )
        surface_mesh.cell_data["viscous_force"] = viscous_data

    pressure_data.index_add_(0, anchor_id, pressure_force)
    viscous_data.index_add_(0, anchor_id, viscous_force)


def _face_forces_from_sample(
    sample: SurfaceSample,
    *,
    rho: float,
    surface_area_vectors: Float[torch.Tensor, " F_surface 3"],
) -> tuple[
    Float[torch.Tensor, " F_kept 3"],
    Float[torch.Tensor, " F_kept 3"],
    Float[torch.Tensor, " F_kept 3"],
    Int[torch.Tensor, " F_kept"],
    Float[torch.Tensor, " F_kept 3"],
] | None:
    if sample.anchor_id.numel() == 0:
        return None

    # Graphlow/STL area vectors are body-outward. The fluid-domain
    # normal for the body boundary points into the body.
    S_body = -surface_area_vectors[sample.anchor_id]
    alignment = torch.sum(sample.n_hat * S_body, dim=1, keepdim=True)
    keep = alignment[:, 0] > 0.0
    if not torch.any(keep):
        return None

    n_surface = S_body / torch.linalg.vector_norm(S_body, dim=1, keepdim=True)
    normal_alignment = torch.sum(sample.n_hat * n_surface, dim=1, keepdim=True)
    wall_dist = sample.mag_d[keep] * normal_alignment[keep].clamp_min(1.0e-12)
    dUdn = (sample.U_b[keep] - sample.U_cell[keep]) / wall_dist
    grad_U = dUdn[:, :, None] * n_surface[keep, None, :]
    viscous_stress = (
        rho
        * sample.nu_eff[keep, :, None]
        * (grad_U + torch.transpose(grad_U, 1, 2))
    )
    pressure_force = rho * sample.p_b[keep] * sample.Sf[keep]
    viscous_force = -torch.matmul(viscous_stress, sample.Sf[keep, :, None]).squeeze(
        -1
    )
    face_force = pressure_force + viscous_force

    return (
        pressure_force,
        viscous_force,
        face_force,
        sample.anchor_id[keep],
        sample.face_centers[keep],
    )


def integrate_patch_on_surface_mesh(
    grid: AxisProjectedGrid,
    *,
    p: CellField,
    U: CellField,
    nu_eff: Float[torch.Tensor, " C 1"],
    patch_name: str,
    rho: float,
    center_of_rotation: Float[torch.Tensor, " 3"],
) -> tuple[Float[torch.Tensor, " 3"], Float[torch.Tensor, " 3"]]:
    """
    Integrate an immersed patch using AP projected surface samples.

    The original surface-mesh normal chooses the physical AP side. The
    force is still integrated over AP projected faces, preserving the
    axis-projected method's surface coverage without a solid/fluid cell
    classification.
    """
    surface_mesh = grid.surface_mesh
    surface_area_vectors = surface_mesh.geometry.face_area_vectors().to(
        dtype=grid.dtype, device=grid.device
    )

    force = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
    moment = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
    CofR = center_of_rotation.to(dtype=grid.dtype, device=grid.device)

    for sample in iter_patch_side_samples(
        grid,
        p=p,
        U=U,
        nu_eff=nu_eff,
        patch_name=patch_name,
    ):
        face_forces = _face_forces_from_sample(
            sample,
            rho=rho,
            surface_area_vectors=surface_area_vectors,
        )
        if face_forces is None:
            continue

        pressure_force, viscous_force, face_force, anchor_id, face_centers = (
            face_forces
        )

        accumulate_sample_forces_on_surface_mesh(
            grid,
            anchor_id=anchor_id,
            pressure_force=pressure_force,
            viscous_force=viscous_force,
        )

        force = force + torch.sum(face_force, dim=0)
        moment_arm = face_centers - CofR
        moment = moment + torch.sum(torch.cross(moment_arm, face_force, dim=1), dim=0)

    return force, moment
