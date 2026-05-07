import logging
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float

from gridfoam.core.equation import Equation

logger = logging.getLogger(__name__)


class LinearSolver(ABC):
    """
    Abstract base class for linear solvers.
    """

    @abstractmethod
    def solve(
        self,
        eq: Equation,
    ) -> Float[torch.Tensor, " C k"]:
        """
        Solve ``A x = b`` and return updated ``x``.

        Parameters
        ----------
        eq : Equation
            Equation object to solve.

        Returns
        -------
        Float[torch.Tensor, " C k"]
            Computed solution tensor.
        """
        pass


def is_converged(
    thresh: Float[torch.Tensor, " 1"],
    norm_res: Float[torch.Tensor, " 1"],
) -> bool:
    """
    Return True if residual norm is below threshold.

    Parameters
    ----------
    thresh : Float[torch.Tensor, " 1"]
        Convergence threshold.
    norm_res : Float[torch.Tensor, " 1"]
        Current residual norm.
    """
    return bool((norm_res <= thresh).item())


def residual_threshold(
    atol: float,
    rtol: float,
    ref_norm: Float[torch.Tensor, " 1"],
) -> Float[torch.Tensor, " 1"]:
    """
    Compute the residual convergence threshold.

    Parameters
    ----------
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    ref_norm : Float[torch.Tensor, " 1"]
        Reference residual norm.

    Returns
    -------
    Float[torch.Tensor, " 1"]
        Residual convergence threshold ``atol + rtol * ref_norm``.
    """
    return atol + rtol * ref_norm
