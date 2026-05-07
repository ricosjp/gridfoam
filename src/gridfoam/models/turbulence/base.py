import logging
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float

from gridfoam.core.builtins import make_builtin_key
from gridfoam.core.field import CellField, FaceField, FieldRole
from gridfoam.core.grid.base import IGridBase

logger = logging.getLogger(__name__)


class TurbulenceModel(ABC):
    """
    Abstract base class for turbulence models.

    Parameters
    ----------
    grid : IGridBase
        Computational grid.
    nu : float or torch.Tensor
        Molecular kinematic viscosity.
    """

    def __init__(self, grid: IGridBase, nu: float | torch.Tensor):
        self.nu = nu
        self.grid = grid
        builtin_scope = "global"

        # Initialize turbulent viscosity field.
        self.nu_t = CellField(
            grid=self.grid,
            name="nu_t",
            role=FieldRole.LOCAL,
            num_components=1,
        )
        self.grid.register_builtin_field(
            make_builtin_key("nu_t", scope=builtin_scope), self.nu_t
        )

    @abstractmethod
    def correct(self, U: CellField, phi: FaceField) -> None:
        """
        Solve turbulence equations and update ``nu_t``.
        """
        pass

    @abstractmethod
    def nu_eff(self) -> Float[torch.Tensor, " C 1"]:
        """
        Return effective viscosity ``nu + nu_t``.

        Returns
        -------
        torch.Tensor
            Effective viscosity per cell with shape ``[C, 1]``.
        """
        pass
