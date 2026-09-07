from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float

from gridfoam.core.equation import Equation
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.solvers.adjoint import attach_implicit_adjoint

GradientMode = Literal["adjoint", "unrolled"]


@dataclass(frozen=True)
class SolveStats:
    """
    Statistics from a single linear-system solve.

    Attributes
    ----------
    solver : str
        Solver type name (for example ``"cg"``).
    initial_residual : float
        Initial algebraic residual norm (not normalized).
    final_residual : float
        Final algebraic residual norm (not normalized).
    iterations : int
        Number of solver iterations performed.
    converged : bool
        Whether convergence criteria were met.
    """

    solver: str
    """Solver type name (for example ``"cg"``)."""

    initial_residual: float
    """Initial algebraic residual norm (not normalized)."""

    final_residual: float
    """Final algebraic residual norm (not normalized)."""

    iterations: int
    """Number of solver iterations performed."""

    converged: bool
    """Whether convergence criteria were met."""


@dataclass(frozen=True)
class SolveResult:
    """
    Solution tensor and associated solver statistics.

    Attributes
    ----------
    solution : torch.Tensor
        Computed field values with shape ``[C, *component_shape]``.
    stats : tuple[SolveStats, ...]
        Per-component solver statistics in equation component order.
    """

    solution: Float[torch.Tensor, " C *component_shape"]
    """Computed field values with shape ``[C, *component_shape]``."""

    stats: tuple[SolveStats, ...]
    """Per-component solver statistics in equation component order."""


class LinearSolver(ABC):
    """
    Abstract base class for linear solvers.

    By default ``solve`` detaches the primal solve and attaches an implicit
    adjoint. Set ``grad_mode="unrolled"`` to differentiate through the
    primal iterations instead (Krylov solvers only).

    Attributes
    ----------
    atol : float
        Absolute residual tolerance.
    rtol : float
        Relative residual tolerance.
    grad_mode : {"adjoint", "unrolled"}
        Differentiation mode for ``solve``.
    """

    atol: float
    """Absolute residual tolerance."""

    rtol: float
    """Relative residual tolerance."""

    grad_mode: GradientMode = "adjoint"
    """Differentiation mode for ``solve``."""

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
        if self.grad_mode == "unrolled":
            return self._solve_primal(eq)

        A = eq.fv_matrix
        with torch.no_grad():
            result = self._solve_primal(eq)
        solution = attach_implicit_adjoint(
            A,
            result.solution,
            solve_transpose=self.solve_transpose,
        )
        return SolveResult(solution=solution, stats=result.stats)

    @abstractmethod
    def _solve_primal(self, eq: Equation) -> SolveResult:
        """Solve ``A x = b`` without attaching an adjoint."""

    @abstractmethod
    def solve_transpose(
        self,
        A_T: FvMatrix,
        rhs: Float[torch.Tensor, " C *component_shape"],
    ) -> Float[torch.Tensor, " C *component_shape"]:
        """Solve ``A_T y = rhs`` (used by the implicit adjoint)."""


def is_converged(
    thresh: Float[torch.Tensor, ""],
    norm_res: Float[torch.Tensor, ""],
) -> bool:
    """
    Return True if residual norm is below threshold.

    Parameters
    ----------
    thresh : Float[torch.Tensor, ""]
        Convergence threshold.
    norm_res : Float[torch.Tensor, ""]
        Current residual norm.
    """
    return bool((norm_res < thresh).item())


def residual_threshold(
    atol: float,
    rtol: float,
    ref_norm: Float[torch.Tensor, ""],
) -> Float[torch.Tensor, ""]:
    """
    Compute the residual convergence threshold.

    Parameters
    ----------
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.
    ref_norm : Float[torch.Tensor, ""]
        Reference residual norm.

    Returns
    -------
    Float[torch.Tensor, ""]
        Residual convergence threshold ``max(atol, rtol * ref_norm)``.
    """
    return torch.maximum(torch.full_like(ref_norm, atol), rtol * ref_norm)
