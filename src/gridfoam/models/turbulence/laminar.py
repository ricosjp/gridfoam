import logging

import torch
from jaxtyping import Float

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.base import IGridBase
from gridfoam.models.turbulence.base import TurbulenceModel

logger = logging.getLogger(__name__)


class Laminar(TurbulenceModel):
    """
    Laminar model.

    Uses only molecular viscosity and does not add turbulent viscosity.
    """

    def __init__(self, grid: IGridBase):
        super().__init__(grid)

    def correct(self, U: CellField, phi: FaceField) -> None:
        """
        No-op for laminar model.
        """
        pass

    def nu_eff(self) -> Float[torch.Tensor, " C 1"]:
        """
        Return effective viscosity ``nu + nu_t``.

        Returns
        -------
        torch.Tensor
            Effective viscosity per cell with shape ``[C, 1]``.
        """
        nu_eff_value = self.transport.nu() + self.nu_t.data
        nu_eff_min = torch.min(nu_eff_value).item()
        nu_eff_max = torch.max(nu_eff_value).item()
        logger.debug(
            "Laminar nu_eff range=[%.3e, %.3e]",
            nu_eff_min,
            nu_eff_max,
        )
        return nu_eff_value
