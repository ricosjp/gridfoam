from __future__ import annotations

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

        immersed_owner = grid.owner[grid.ap_is_immersed_faces]
        immersed_neighbour = grid.neighbour[grid.ap_is_immersed_faces]
        immersed_Sf = grid.Sf[grid.ap_is_immersed_faces]

        for patch_name in self.patches:
            upper_mask, lower_mask = grid.ap_get_patch_mask(patch_name)

            if torch.any(upper_mask):
                # Upper faces use the owner-side AP face normal.
                Sf = immersed_Sf[upper_mask]
                f_patch, m_patch = self._integrate_side(
                    grid=grid,
                    p=p,
                    U=U,
                    nu_eff=nu_eff,
                    patch_name=patch_name,
                    side=FaceSide.UPPER,
                    target_cells=immersed_owner[upper_mask],
                    Sf=Sf,
                    mag_d=grid.ap_dist_owner_to_bnd[upper_mask],
                )
                force = force + f_patch
                moment = moment + m_patch

            if torch.any(lower_mask):
                # Lower faces use the opposite AP face normal.
                Sf = -immersed_Sf[lower_mask]
                f_patch, m_patch = self._integrate_side(
                    grid=grid,
                    p=p,
                    U=U,
                    nu_eff=nu_eff,
                    patch_name=patch_name,
                    side=FaceSide.LOWER,
                    target_cells=immersed_neighbour[lower_mask],
                    Sf=Sf,
                    mag_d=grid.ap_dist_neighbour_to_bnd[lower_mask],
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

    def _integrate_side(
        self,
        grid: AxisProjectedGrid,
        p: CellField,
        U: CellField,
        nu_eff: Float[torch.Tensor, " C 1"],
        patch_name: str,
        side: FaceSide,
        target_cells: Int[torch.Tensor, " F_patch"],
        Sf: Float[torch.Tensor, " F_patch 3"],
        mag_d: Float[torch.Tensor, " F_patch 1"],
    ) -> tuple[Float[torch.Tensor, " 3"], Float[torch.Tensor, " 3"]]:
        """
        Integrate force and moment on one side of an AP immersed patch.

        Parameters
        ----------
        grid : AxisProjectedGrid
            Grid containing geometry and surface mesh references.
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

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Integrated force and moment vectors.
        """
        p_b = self._boundary_value(p, patch_name, side, target_cells, mag_d)
        U_b = self._boundary_value(U, patch_name, side, target_cells, mag_d)

        mag_Sf = torch.linalg.vector_norm(Sf, dim=1, keepdim=True)
        n_hat = Sf / mag_Sf

        # ``Sf`` points outward from the fluid cell into the body.
        # With kinematic pressure, the force on the body is +rho*p*Sf.
        pressure_force = self.rho * p_b * Sf

        # ``tau@Sf`` is the traction exerted by the body on the fluid. The
        # force on the body has the opposite sign.
        dUdn = (U_b - U.data[target_cells]) / mag_d  # [F_patch 3]
        grad_U = dUdn[:, :, None] * n_hat[:, None, :]  # [F_patch 3 3]
        viscous_stress = (
            self.rho
            * nu_eff[target_cells, :, None]
            * (grad_U + torch.transpose(grad_U, 1, 2))
        )
        viscous_force = -torch.matmul(viscous_stress, Sf[:, :, None]).squeeze(
            -1
        )  # fij, fj -> fi

        # just for visualizing the force
        self._map_force_to_surface_mesh(
            grid=grid,
            patch_name=patch_name,
            side=side,
            pressure_force=pressure_force,
            viscous_force=viscous_force,
        )

        face_force = pressure_force + viscous_force
        force = torch.sum(face_force, dim=0)

        CofR = self.CofR.to(dtype=grid.dtype, device=grid.device)
        face_centers = grid.cell_centers[target_cells] + n_hat * mag_d
        moment_arm = face_centers - CofR
        moment = torch.sum(torch.cross(moment_arm, face_force, dim=1), dim=0)

        return force, moment

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

    def _map_force_to_surface_mesh(
        self,
        grid: AxisProjectedGrid,
        patch_name: str,
        side: FaceSide,
        pressure_force: Float[torch.Tensor, " F_patch 3"],
        viscous_force: Float[torch.Tensor, " F_patch 3"],
    ) -> None:
        """
        Accumulate projected-face forces onto their anchor surface faces.

        Multiple AP-projected faces may map to the same surface face. The
        mapped value is therefore a resultant force per surface face.

        Parameters
        ----------
        grid : AxisProjectedGrid
            Grid owning the surface mesh and anchor-id arrays.
        patch_name : str
            Immersed patch name.
        side : FaceSide
            Owner-side or neighbour-side boundary.
        pressure_force : torch.Tensor
            Pressure-force contribution on AP-projected faces.
        viscous_force : torch.Tensor
            Viscous-force contribution on AP-projected faces.
        """
        surface_mesh = grid.surface_mesh
        upper_mask, lower_mask = grid.ap_get_patch_mask(patch_name)
        match side:
            case FaceSide.UPPER:
                anchor_id = grid.ap_owner_bnd_anchor_id[upper_mask]
            case FaceSide.LOWER:
                anchor_id = grid.ap_neighbour_bnd_anchor_id[lower_mask]
            case _:
                raise ValueError(f"Invalid side: {side}")

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
