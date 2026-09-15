import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float

from gridfoam.core.equation import Equation
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.solvers.adjoint import attach_implicit_adjoint

GradientMode = Literal["adjoint", "unrolled"]


class LinearSolveError(RuntimeError):
    """A required primal or transpose linear solve failed to converge."""


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

    require_convergence: bool = False
    """Raise on failed solves; captured by each strict implicit backward."""

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
            result = self._solve_primal(eq)
            self._check_result(result, eq.name, "Primal")
            return result

        A = eq.fv_matrix
        with torch.no_grad():
            result = self._solve_primal(eq)
        self._check_result(result, eq.name, "Primal")
        # The caller may restore its solver policy before backward, or run
        # another forward. Keep this solve's strict policy and scalar settings.
        transpose = (
            copy(self).solve_transpose
            if self.require_convergence
            else self.solve_transpose
        )
        solution = attach_implicit_adjoint(
            A,
            result.solution,
            solve_transpose=transpose,
        )
        return SolveResult(solution=solution, stats=result.stats)

    def _check_result(
        self, result: SolveResult, field_name: str, stage: str
    ) -> None:
        """Propagate the solver's per-component convergence decision."""
        if not self.require_convergence:
            return
        for component, stats in enumerate(result.stats):
            if not stats.converged or not math.isfinite(stats.final_residual):
                raise LinearSolveError(
                    f"{stage} solve failed for {field_name!r}, component "
                    f"{component}: residual={stats.final_residual}, "
                    f"iterations={stats.iterations}"
                )
        if not bool(torch.isfinite(result.solution).all()):
            raise LinearSolveError(
                f"Non-finite {stage.lower()} solution for {field_name!r}"
            )

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


@contextmanager
def require_converged_solves(
    solvers: Iterable[LinearSolver],
) -> Generator[None]:
    """Require successful solves in this scope, including later backward.

    Solver instances must not be used concurrently. Implicit backward keeps
    the strict policy after this scope has restored the caller's settings.
    """
    saved = {solver: solver.require_convergence for solver in solvers}
    try:
        for solver in saved:
            solver.require_convergence = True
        yield
    finally:
        for solver, previous in saved.items():
            solver.require_convergence = previous
