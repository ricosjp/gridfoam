from __future__ import annotations

import csv
from pathlib import Path

import torch

from gridfoam.core.field import CellField, get_or_create_cellfield
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.name import make_field_name
from gridfoam.meta.enums import FieldRole
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.post.forces.coeffs import ForceCoeffs
from gridfoam.post.forces.coord import OrthonormalCoord
from gridfoam.post.forces.integration import (
    integrate_patch_on_surface_mesh,
    reset_surface_force_data,
)


class ForceEvaluator:
    """
    Evaluate aerodynamic force and moment coefficients on immersed surfaces.
    """

    def __init__(self, grid: IGridBase, phase: str | None = None):
        self.grid = grid

        post_processing = grid.sim_config.post_processing
        assert post_processing is not None
        config = post_processing.forceCoeff
        assert config is not None
        self.patches = config.patches
        self.rho = config.rho
        self.magU_ref = config.magU_ref
        self.A_ref = config.A_ref
        self.L_ref = config.L_ref
        self.local_coord = OrthonormalCoord.from_local_coord(config.local_coord)
        self.CofR = torch.tensor(config.local_coord.center_of_rotation)

        U_name = make_field_name("U", phase=phase)
        p_name = make_field_name("p", phase=phase)
        # Reuse the solver's existing U/p fields regardless of their role
        # (SIMPLE registers U as LOCAL, PIMPLE as TRANSIENT). Fall back to
        # creating LOCAL fields when they do not exist yet.
        self.U = self._resolve_field(grid, U_name, 3)
        self.p = self._resolve_field(grid, p_name, 1)

        self.history: list[ForceCoeffs] = []

    @staticmethod
    def _resolve_field(
        grid: IGridBase,
        name: str,
        num_components: int,
    ) -> CellField:
        """Return the existing cell field or create a LOCAL one if absent."""
        field = grid.get_cellfield(name)
        if field is not None:
            return field
        return get_or_create_cellfield(
            grid, name, FieldRole.LOCAL, num_components
        )

    def evaluate(
        self,
        grid: IGridBase,
        time: float,
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
        turbulence : TurbulenceModel
            Turbulence model that provides the effective kinematic viscosity.

        Returns
        -------
        ForceCoeffs
            Force and moment coefficients at ``time``.
        """
        assert isinstance(grid, AxisProjectedGrid)
        reset_surface_force_data(grid)

        force_scale, moment_scale = self._reference_scales()
        nu_eff = turbulence.nu_eff()

        force = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
        moment = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
        for patch_name in self.patches:
            f_patch, m_patch = integrate_patch_on_surface_mesh(
                grid,
                p=self.p,
                U=self.U,
                nu_eff=nu_eff,
                patch_name=patch_name,
                rho=self.rho,
                center_of_rotation=self.CofR,
            )
            force = force + f_patch
            moment = moment + m_patch

        coord = self.local_coord.on_device(dtype=grid.dtype, device=grid.device)
        coeffs = ForceCoeffs.from_force_moment(
            time=time,
            force=force,
            moment=moment,
            coord=coord,
            force_scale=force_scale,
            moment_scale=moment_scale,
        )

        self.history.append(coeffs)
        return coeffs

    def get_history(self) -> list[ForceCoeffs]:
        return self.history

    def write_csv(self, path: Path) -> None:
        with path.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time",
                    "Cd",
                    "Cd_f",
                    "Cd_r",
                    "Cl",
                    "Cl_f",
                    "Cl_r",
                    "CmPitch",
                    "CmRoll",
                    "CmYaw",
                    "Cs",
                    "Cs_f",
                    "Cs_r",
                ]
            )
            for coeff in self.history:
                writer.writerow(
                    [
                        coeff.time,
                        coeff.Cd.item(),
                        coeff.Cd_f.item(),
                        coeff.Cd_r.item(),
                        coeff.Cl.item(),
                        coeff.Cl_f.item(),
                        coeff.Cl_r.item(),
                        coeff.CmPitch.item(),
                        coeff.CmRoll.item(),
                        coeff.CmYaw.item(),
                        coeff.Cs.item(),
                        coeff.Cs_f.item(),
                        coeff.Cs_r.item(),
                    ]
                )

    def save_surface_mesh(self, path: Path) -> None:
        self.grid.surface_mesh.save(
            path,
            overwrite_features=True,
            overwrite_file=True,
        )

    def _reference_scales(self) -> tuple[float, float]:
        q_inf = 0.5 * self.rho * self.magU_ref**2
        force_scale = q_inf * self.A_ref
        moment_scale = force_scale * self.L_ref
        if force_scale == 0.0 or moment_scale == 0.0:
            raise ValueError("force coefficient reference scale is zero")
        return force_scale, moment_scale
