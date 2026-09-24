import logging

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.base import GridBase
from gridfoam.models.turbulence.base import TurbulenceModel

logger = logging.getLogger(__name__)


class Laminar(TurbulenceModel):
    """
    Laminar model.

    Uses only molecular viscosity and does not add turbulent viscosity.
    """

    def __init__(self, grid: GridBase):
        super().__init__(grid)

    def correct(self, U: CellField, phi: FaceField) -> None:
        """
        No-op for laminar model.
        """
        pass

    def nu_eff(self) -> Float[torch.Tensor, " C"]:
        """
        Return effective viscosity ``nu + nu_t``.

        Returns
        -------
        torch.Tensor
            Effective viscosity per cell with shape ``[C]``.
        """
        nu_eff_value = self.transport.nu() + self.nu_t.data
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Laminar nu_eff range=[%.3e, %.3e]",
                torch.min(nu_eff_value).item(),
                torch.max(nu_eff_value).item(),
            )
        return nu_eff_value
