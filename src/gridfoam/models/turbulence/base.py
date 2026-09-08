import logging
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float

from gridfoam.core.dimensions import DIM_NU
from gridfoam.core.field import (
    CellField,
    FaceField,
    FieldRole,
    get_or_create_cellfield,
)
from gridfoam.core.grid.base import GridBase
from gridfoam.core.name import make_field_name
from gridfoam.models.transport.base import TransportModel
from gridfoam.models.transport.factory import create_transport_model

logger = logging.getLogger(__name__)


class TurbulenceModel(ABC):
    """
    Abstract base class for turbulence models.

    Parameters
    ----------
    grid : GridBase
        Computational grid.

    Attributes
    ----------
    grid : GridBase
        Computational grid.
    transport : TransportModel
        Molecular transport model providing ``nu``.
    nu_t : CellField
        Turbulent kinematic viscosity with shape ``[C]``.
    """

    grid: GridBase
    """Computational grid."""

    transport: TransportModel
    """Molecular transport model providing ``nu``."""

    nu_t: CellField
    """Turbulent kinematic viscosity with shape ``[C]``."""

    def __init__(self, grid: GridBase):
        self.grid = grid
        self.transport = create_transport_model(grid)

        # Initialize turbulent viscosity field.
        nu_t_name = make_field_name("nu_t")
        self.nu_t = get_or_create_cellfield(
            self.grid, nu_t_name, FieldRole.LOCAL, (), dimension=DIM_NU
        )

    @abstractmethod
    def correct(self, U: CellField, phi: FaceField) -> None:
        """
        Solve turbulence equations and update ``nu_t``.
        """
        pass

    @abstractmethod
    def nu_eff(self) -> Float[torch.Tensor, " C"]:
        """
        Return effective viscosity ``nu + nu_t``.

        Returns
        -------
        torch.Tensor
            Effective viscosity per cell with shape ``[C]``.
        """
        pass
