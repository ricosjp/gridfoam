from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.config import (
    DragLiftCoord,
    DragPitchCoord,
    ForceCoeffConfig,
    ForceCoord,
)
from gridfoam.meta.enums import FaceSide, ForceCoordMode
from gridfoam.models.turbulence.base import TurbulenceModel


def _normalize_vector(
    arr: Float[torch.Tensor, " 3"],
) -> Float[torch.Tensor, " 3"]:
    norm = torch.linalg.vector_norm(arr)
    if norm == 0:
        raise ValueError("vector is zero vector")
    return arr / norm


@dataclass(frozen=True)
class OrthonormalCoord:
    """
    Orthonormal basis used for force and moment coefficients.

    Attributes
    ----------
    e1 : torch.Tensor
        Drag and roll axis.
    e2 : torch.Tensor
        Side-force and pitch axis.
    e3 : torch.Tensor
        Lift and yaw axis.
    """

    e1: Float[torch.Tensor, " 3"]  # drag / roll
    e2: Float[torch.Tensor, " 3"]  # side / pitch
    e3: Float[torch.Tensor, " 3"]  # lift / yaw

    @classmethod
    def from_local_coord(cls, local_coord: ForceCoord) -> OrthonormalCoord:
        match local_coord.mode:
            case ForceCoordMode.DRAG_LIFT:
                assert isinstance(local_coord, DragLiftCoord)
                e1 = _normalize_vector(torch.tensor(local_coord.drag_dir))
                lift = torch.tensor(local_coord.lift_dir)
                e3_raw = lift - e1 * (lift @ e1)
                e3 = _normalize_vector(e3_raw)
                e2 = torch.linalg.cross(e3, e1)

            case ForceCoordMode.DRAG_PITCH:
                assert isinstance(local_coord, DragPitchCoord)
                e1 = _normalize_vector(torch.tensor(local_coord.drag_dir))
                pitch = torch.tensor(local_coord.pitch_axis)
                e2_raw = pitch - e1 * (pitch @ e1)
                e2 = _normalize_vector(e2_raw)
                e3 = torch.linalg.cross(e1, e2)
        return cls(
            e1=e1,
            e2=e2,
            e3=e3,
        )


@dataclass(frozen=True)
class ForceCoeffs:
    """
    Aerodynamic force and moment coefficients at one time.

    Attributes
    ----------
    time : float
        Simulation time.
    Cd, Cs, Cl : torch.Tensor
        Drag, side-force and lift coefficients.
    CmRoll, CmPitch, CmYaw : torch.Tensor
        Moment coefficients about the local roll, pitch and yaw axes.
    Cd_f, Cd_r, Cs_f, Cs_r, Cl_f, Cl_r : torch.Tensor
        OpenFOAM-style front and rear axle constituents.
    """

    time: float
    Cd: Float[torch.Tensor, " 1"]
    Cd_f: Float[torch.Tensor, " 1"]
    Cd_r: Float[torch.Tensor, " 1"]
    Cl: Float[torch.Tensor, " 1"]
    Cl_f: Float[torch.Tensor, " 1"]
    Cl_r: Float[torch.Tensor, " 1"]
    CmPitch: Float[torch.Tensor, " 1"]
    CmRoll: Float[torch.Tensor, " 1"]
    CmYaw: Float[torch.Tensor, " 1"]
    Cs: Float[torch.Tensor, " 1"]
    Cs_f: Float[torch.Tensor, " 1"]
    Cs_r: Float[torch.Tensor, " 1"]


class ForceEvaluator:
    """
    Evaluate aerodynamic force and moment coefficients on immersed surfaces.
    """

    def __init__(self, config: ForceCoeffConfig):
        self.patches = config.patches
        self.rho = config.rho
        self.magU_ref = config.magU_ref
        self.A_ref = config.A_ref
        self.L_ref = config.L_ref
        self.local_coord = OrthonormalCoord.from_local_coord(config.local_coord)
        self.CofR = torch.tensor(config.local_coord.center_of_rotation)

        self.history: list[ForceCoeffs] = []

    def evaluate(
        self,
        grid: IGridBase,
        time: float,
        p: CellField,
        U: CellField,
        turbulence: TurbulenceModel,
    ) -> ForceCoeffs:
        """
        Integrate surface forces and return nondimensional coefficients.

        Parameters
        ----------
        grid : IGridBase
            Axis-projected grid containing immersed-boundary metadata.
        time : float
            Simulation time.
        p : CellField
            Kinematic pressure field.
        U : CellField
            Velocity field.
        turbulence : TurbulenceModel
            Turbulence model that provides the effective kinematic viscosity.

        Returns
        -------
        ForceCoeffs
            Force and moment coefficients at ``time``.
        """
        assert isinstance(grid, AxisProjectedGrid)
        self._reset_surface_force_data(grid)

        q_inf = 0.5 * self.rho * self.magU_ref**2
        force_scale = q_inf * self.A_ref
        moment_scale = force_scale * self.L_ref
        if force_scale == 0.0 or moment_scale == 0.0:
            raise ValueError("force coefficient reference scale is zero")

        force = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
        moment = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
        nu_eff = turbulence.nu_eff()

        for patch_name in self.patches:
            f_patch, m_patch = self._integrate_patch_on_surface_mesh(
                grid=grid,
                p=p,
                U=U,
                nu_eff=nu_eff,
                patch_name=patch_name,
            )
            force = force + f_patch
            moment = moment + m_patch

        e1 = self.local_coord.e1.to(dtype=grid.dtype, device=grid.device)
        e2 = self.local_coord.e2.to(dtype=grid.dtype, device=grid.device)
        e3 = self.local_coord.e3.to(dtype=grid.dtype, device=grid.device)

        Cd = torch.sum(force * e1, dim=0, keepdim=True) / force_scale
        Cs = torch.sum(force * e2, dim=0, keepdim=True) / force_scale
        Cl = torch.sum(force * e3, dim=0, keepdim=True) / force_scale
        CmRoll = torch.sum(moment * e1, dim=0, keepdim=True) / moment_scale
        CmPitch = torch.sum(moment * e2, dim=0, keepdim=True) / moment_scale
        CmYaw = torch.sum(moment * e3, dim=0, keepdim=True) / moment_scale

        coeffs = ForceCoeffs(
            time=time,
            Cd=Cd,
            Cd_f=0.5 * Cd + CmRoll,
            Cd_r=0.5 * Cd - CmRoll,
            Cl=Cl,
            Cl_f=0.5 * Cl + CmPitch,
            Cl_r=0.5 * Cl - CmPitch,
            CmPitch=CmPitch,
            CmRoll=CmRoll,
            CmYaw=CmYaw,
            Cs=Cs,
            Cs_f=0.5 * Cs + CmYaw,
            Cs_r=0.5 * Cs - CmYaw,
        )

        self.history.append(coeffs)

        return coeffs

    def get_history(self) -> list[ForceCoeffs]:
        return self.history

    def _integrate_patch_on_surface_mesh(
        self,
        grid: AxisProjectedGrid,
        p: CellField,
        U: CellField,
        nu_eff: Float[torch.Tensor, " C 1"],
        patch_name: str,
    ) -> tuple[Float[torch.Tensor, " 3"], Float[torch.Tensor, " 3"]]:
        """
        Integrate an immersed patch using AP projected surface samples.

        The original surface-mesh normal chooses the physical AP side. The
        force is still integrated over AP projected faces, preserving the
        axis-projected method's surface coverage without a solid/fluid cell
        classification.
        """
        surface_mesh = grid.surface_mesh
        surface_S_out = surface_mesh.geometry.face_area_vectors().to(
            dtype=grid.dtype, device=grid.device
        )

        force = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
        moment = torch.zeros((3,), dtype=grid.dtype, device=grid.device)

        for sample in self._iter_side_samples(
            grid=grid,
            p=p,
            U=U,
            nu_eff=nu_eff,
            patch_name=patch_name,
        ):
            if sample.anchor_id.numel() == 0:
                continue

            # Graphlow/STL area vectors are body-outward. The fluid-domain
            # normal for the body boundary points into the body.
            S_body = -surface_S_out[sample.anchor_id]
            alignment = torch.sum(sample.n_hat * S_body, dim=1, keepdim=True)
            keep = alignment[:, 0] > 0.0
            if not torch.any(keep):
                continue

            n_surface = S_body / torch.linalg.vector_norm(
                S_body, dim=1, keepdim=True
            )
            normal_alignment = torch.sum(
                sample.n_hat * n_surface, dim=1, keepdim=True
            )
            wall_dist = sample.mag_d[keep] * normal_alignment[keep].clamp_min(
                1.0e-12
            )
            dUdn = (sample.U_b[keep] - sample.U_cell[keep]) / wall_dist
            grad_U = dUdn[:, :, None] * n_surface[keep, None, :]
            viscous_stress = (
                self.rho
                * sample.nu_eff[keep, :, None]
                * (grad_U + torch.transpose(grad_U, 1, 2))
            )
            pressure_force = self.rho * sample.p_b[keep] * sample.Sf[keep]
            viscous_force = -torch.matmul(
                viscous_stress, sample.Sf[keep, :, None]
            ).squeeze(-1)
            face_force = pressure_force + viscous_force

            self._map_sample_force_to_surface_mesh(
                grid=grid,
                anchor_id=sample.anchor_id[keep],
                pressure_force=pressure_force,
                viscous_force=viscous_force,
            )

            force = force + torch.sum(face_force, dim=0)
            CofR = self.CofR.to(dtype=grid.dtype, device=grid.device)
            moment_arm = sample.face_centers[keep] - CofR
            moment = moment + torch.sum(
                torch.cross(moment_arm, face_force, dim=1), dim=0
            )

        return force, moment

    def _iter_side_samples(
        self,
        grid: AxisProjectedGrid,
        p: CellField,
        U: CellField,
        nu_eff: Float[torch.Tensor, " C 1"],
        patch_name: str,
    ) -> Iterable[_SurfaceSample]:
        immersed_owner = grid.owner[grid.ap_is_immersed_faces]
        immersed_neighbour = grid.neighbour[grid.ap_is_immersed_faces]
        immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]
        upper_mask, lower_mask = grid.ap_get_patch_mask(patch_name)

        if torch.any(upper_mask):
            yield self._sample_side(
                p=p,
                U=U,
                nu_eff=nu_eff,
                patch_name=patch_name,
                side=FaceSide.UPPER,
                target_cells=immersed_owner[upper_mask],
                Sf=immersed_Sf[upper_mask],
                mag_d=grid.ap_dist_owner_to_bnd[upper_mask],
                anchor_id=grid.ap_owner_bnd_anchor_id[upper_mask],
                cell_centers=grid.cell_centers[immersed_owner[upper_mask]],
            )

        if torch.any(lower_mask):
            yield self._sample_side(
                p=p,
                U=U,
                nu_eff=nu_eff,
                patch_name=patch_name,
                side=FaceSide.LOWER,
                target_cells=immersed_neighbour[lower_mask],
                Sf=-immersed_Sf[lower_mask],
                mag_d=grid.ap_dist_neighbour_to_bnd[lower_mask],
                anchor_id=grid.ap_neighbour_bnd_anchor_id[lower_mask],
                cell_centers=grid.cell_centers[immersed_neighbour[lower_mask]],
            )

    def _sample_side(
        self,
        p: CellField,
        U: CellField,
        nu_eff: Float[torch.Tensor, " C 1"],
        patch_name: str,
        side: FaceSide,
        target_cells: Int[torch.Tensor, " F_patch"],
        Sf: Float[torch.Tensor, " F_patch 3"],
        mag_d: Float[torch.Tensor, " F_patch 1"],
        anchor_id: Int[torch.Tensor, " F_patch"],
        cell_centers: Float[torch.Tensor, " F_patch 3"],
    ) -> _SurfaceSample:
        """
        Evaluate AP-side boundary samples for later surface-face integration.

        Parameters
        ----------
        p : CellField
            Kinematic pressure field.
        U : CellField
            Velocity field.
        nu_eff : torch.Tensor
            Effective kinematic viscosity per cell.
        patch_name : str
            Immersed patch name.
        side : FaceSide
            Owner-side or neighbour-side boundary.
        target_cells : torch.Tensor
            Fluid cells adjacent to the immersed boundary faces.
        Sf : torch.Tensor
            AP-projected face area vectors.
        mag_d : torch.Tensor
            AP-axis distance from cell center to boundary.
        anchor_id : torch.Tensor
            Surface-mesh face ids associated with these AP samples.
        cell_centers : torch.Tensor
            Adjacent-cell centers for AP boundary-point reconstruction.
        """
        p_b = self._boundary_value(p, patch_name, side, target_cells, mag_d)
        U_b = self._boundary_value(U, patch_name, side, target_cells, mag_d)

        mag_Sf = torch.linalg.vector_norm(Sf, dim=1, keepdim=True)
        n_hat = Sf / mag_Sf

        return _SurfaceSample(
            anchor_id=anchor_id,
            n_hat=n_hat,
            Sf=Sf,
            p_b=p_b,
            U_b=U_b,
            U_cell=U.data[target_cells],
            nu_eff=nu_eff[target_cells],
            mag_d=mag_d,
            face_centers=cell_centers + n_hat * mag_d,
        )

    def _boundary_value(
        self,
        field: CellField,
        patch_name: str,
        side: FaceSide,
        target_cells: Int[torch.Tensor, " F_patch"],
        mag_d: Float[torch.Tensor, " F_patch 1"],
    ) -> Float[torch.Tensor, " F_patch k"]:
        """
        Evaluate a boundary value using the field's value-fraction form.

        Parameters
        ----------
        field : CellField
            Field whose boundary state is requested.
        patch_name : str
            Patch name.
        side : FaceSide
            AP-IBM face side.
        target_cells : torch.Tensor
            Adjacent fluid-cell ids.
        mag_d : torch.Tensor
            Distance from cell center to boundary along the AP axis.

        Returns
        -------
        torch.Tensor
            Boundary values on the requested patch side.
        """
        if patch_name not in field.bcs:
            return field.data[target_cells]

        fraction, ref_v, ref_g = field.bcs[patch_name].evaluate(
            field, patch_name, side=side
        )
        psi_O = field.data[target_cells]
        return fraction * ref_v + (1.0 - fraction) * (psi_O + ref_g * mag_d)

    def _reset_surface_force_data(self, grid: AxisProjectedGrid) -> None:
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

    def _map_sample_force_to_surface_mesh(
        self,
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


@dataclass(frozen=True)
class _SurfaceSample:
    """AP-side sampled boundary data anchored to surface-mesh faces."""

    anchor_id: Int[torch.Tensor, " F_sample"]
    n_hat: Float[torch.Tensor, " F_sample 3"]
    Sf: Float[torch.Tensor, " F_sample 3"]
    p_b: Float[torch.Tensor, " F_sample 1"]
    U_b: Float[torch.Tensor, " F_sample 3"]
    U_cell: Float[torch.Tensor, " F_sample 3"]
    nu_eff: Float[torch.Tensor, " F_sample 1"]
    mag_d: Float[torch.Tensor, " F_sample 1"]
    face_centers: Float[torch.Tensor, " F_sample 3"]
