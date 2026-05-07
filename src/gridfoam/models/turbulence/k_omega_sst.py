import logging

import torch
from jaxtyping import Float

from gridfoam.core.builtins import make_builtin_key
from gridfoam.core.field import CellField, FaceField, FieldRole
from gridfoam.core.grid.base import IGridBase
from gridfoam.models.turbulence.base import TurbulenceModel

logger = logging.getLogger(__name__)


class KOmegaSST(TurbulenceModel):
    """
    k-omega SST (Shear Stress Transport) turbulence model.

    Parameters
    ----------
    grid : IGridBase
        Computational grid.
    nu : float or torch.Tensor
        Kinematic viscosity.
    """

    def __init__(self, grid: IGridBase, nu: float | torch.Tensor):
        super().__init__(grid, nu)
        builtin_scope = "global"

        self.k = CellField(
            grid=self.grid,
            name="k",
            role=FieldRole.LOCAL,
            num_components=1,
        )
        self.grid.register_builtin_field(
            make_builtin_key("k", scope=builtin_scope), self.k
        )
        self.omega = CellField(
            grid=self.grid,
            name="omega",
            role=FieldRole.TRANSIENT,
            num_components=1,
        )
        self.grid.register_builtin_field(
            make_builtin_key("omega", scope=builtin_scope), self.omega
        )

        # Standard k-omega SST constants.
        self.a1 = 0.31
        self.beta_star = 0.09

    def correct(self, U: CellField, phi: FaceField) -> None:
        """
        Solve k and omega transport equations and update ``nu_t``.

        Notes
        -----
        This is currently a prototype placeholder for architecture wiring.
        """
        raise NotImplementedError("KOmegaSST is not implemented yet.")
        # grid = self.grid
        # device = self.grid.device

        # # 1. Compute production from velocity gradient.
        # grad_U = grad(U)
        # # S_mag = ... (compute strain-rate magnitude, etc.)

        # # 2. Solve k equation (assembled from fvm.ddt/div/laplacian).
        # # k_eqn = equation(
        # #     "k", self.k, fvm.div(...) - fvm.laplacian(...) == P_k - Y_k
        # # )

        # # 3. Solve omega equation.
        # # omega_eqn = equation("omega", self.omega, ...)

        # # 4. Update turbulent viscosity nu_t.
        # # In full SST, this uses wall distance and blending functions.

        # # Temporary mock update for executable plumbing.
        # k_val = torch.clamp(self.k.data, min=1e-8)
        # omega_val = torch.clamp(self.omega.data, min=1e-5)

        # self.nu_t.data = k_val / omega_val

    def nu_eff(self) -> Float[torch.Tensor, " C 1"]:
        """
        Return effective viscosity ``nu + nu_t``.

        Returns
        -------
        torch.Tensor
            Effective viscosity per cell with shape ``[C, 1]``.
        """
        nu_eff_value = self.nu + self.nu_t.data
        nu_eff_min = torch.min(nu_eff_value).item()
        nu_eff_max = torch.max(nu_eff_value).item()
        logger.debug(
            "KOmegaSST nu_eff range=[%.3e, %.3e]",
            nu_eff_min,
            nu_eff_max,
        )
        return nu_eff_value
