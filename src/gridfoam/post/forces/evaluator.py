from __future__ import annotations

import torch

from gridfoam.core.field import CellField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.config import ForceCoeffConfig
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
        reset_surface_force_data(grid)

        force_scale, moment_scale = self._reference_scales()
        nu_eff = turbulence.nu_eff()

        force = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
        moment = torch.zeros((3,), dtype=grid.dtype, device=grid.device)
        for patch_name in self.patches:
            f_patch, m_patch = integrate_patch_on_surface_mesh(
                grid,
                p=p,
                U=U,
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

    def _reference_scales(self) -> tuple[float, float]:
        q_inf = 0.5 * self.rho * self.magU_ref**2
        force_scale = q_inf * self.A_ref
        moment_scale = force_scale * self.L_ref
        if force_scale == 0.0 or moment_scale == 0.0:
            raise ValueError("force coefficient reference scale is zero")
        return force_scale, moment_scale
