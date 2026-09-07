from __future__ import annotations

import logging

import torch
from jaxtyping import Float

from gridfoam.core.grid.base import IGridBase
from gridfoam.core.shapes import require_shape
from gridfoam.models.transport.base import TransportModel

logger = logging.getLogger(__name__)


class NewtonianTransport(TransportModel):
    """
    Constant-viscosity Newtonian transport model.

    The case-file value ``properties.transport.nu`` is stored as a tensor
    of shape ``()`` on the grid device. Replace it with ``set_nu`` to
    keep an autograd leaf (including ``nn.Parameter``) in the graph.

    Parameters
    ----------
    grid : IGridBase
        Computational grid that owns device, dtype, and simulator config.
    """

    _nu: Float[torch.Tensor, ""]

    def __init__(self, grid: IGridBase):
        super().__init__(grid)
        self.set_nu(float(grid.sim_config.properties.transport.nu))

    def nu(self) -> Float[torch.Tensor, ""]:
        """
        Return molecular kinematic viscosity.

        The stored tensor is moved to the runtime grid device and dtype.
        When those already match, the same object is returned so autograd
        leaves are preserved.

        Returns
        -------
        torch.Tensor
            Kinematic viscosity ``nu`` with shape ``()``.
        """
        return self._nu.to(dtype=self.grid.dtype, device=self.grid.device)

    def set_nu(self, nu: float | Float[torch.Tensor, ""]) -> None:
        """
        Replace the stored kinematic viscosity.

        A tensor is stored as-is so that ``nn.Parameter`` identity is
        kept. A float is wrapped as a tensor of shape ``()`` on the
        grid device.

        Parameters
        ----------
        nu : float or torch.Tensor
            New kinematic viscosity. Tensors must have shape ``()``.
        """
        if isinstance(nu, torch.Tensor):
            require_shape(nu, (), "uniform viscosity")
            self._nu = nu
            return
        self._nu = torch.tensor(
            float(nu),
            dtype=self.grid.dtype,
            device=self.grid.device,
        )
