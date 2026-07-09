import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from jaxtyping import Float

from gridfoam.core.equation import Equation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolveStats:
    """
    Statistics from a single linear-system solve.

    Parameters
    ----------
    solver : str
        Solver type name (for example ``"cg"``).
    initial_residual : float
        Normalized initial residual norm.
    final_residual : float
        Normalized final residual norm.
    iterations : int
        Number of solver iterations performed.
    converged : bool
        Whether convergence criteria were met.
    """

    solver: str
    initial_residual: float
    final_residual: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class SolveResult:
    """
    Solution tensor and associated solver statistics.

    Parameters
    ----------
    solution : torch.Tensor
        Computed field values with shape ``[C, k]``.
    stats : tuple[SolveStats, ...]
        Per-component solver statistics in equation component order.
    """

    solution: Float[torch.Tensor, " C k"]
    stats: tuple[SolveStats, ...]


class LinearSolver(ABC):
    """
    Abstract base class for linear solvers.
    """

    @abstractmethod
    def solve(
        self,
        eq: Equation,
    ) -> SolveResult:
        """
        Solve ``A x = b`` and return the solution with statistics.

        Parameters
        ----------
        eq : Equation
            Equation object to solve.

        Returns
        -------
        SolveResult
            Computed solution and solver statistics.
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
