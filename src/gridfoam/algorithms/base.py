from abc import ABC, abstractmethod

from gridfoam.core.field import FaceField
from gridfoam.core.grid.base import IGridBase
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.post.diagnostics import DiagnosticsCollector
from gridfoam.solvers.base import GradientMode, LinearSolver, SolveStats


class AlgorithmBase(ABC):
    """
    Abstract base class for CFD algorithms (macro solvers).

    Attributes
    ----------
    grid : IGridBase
        Computational grid owned by the algorithm.
    turbulence : TurbulenceModel
        Turbulence model used to evaluate effective viscosity.
    solvers : dict[str, LinearSolver]
        Linear solvers keyed by field name from ``fvSolution``.
    """

    _diagnostics: DiagnosticsCollector | None = None
    _diagnostics_step: int = 0

    @property
    @abstractmethod
    def grid(self) -> IGridBase:
        """Computational grid owned by the algorithm."""
        pass

    @property
    @abstractmethod
    def turbulence(self) -> TurbulenceModel:
        """Turbulence model used to evaluate effective viscosity."""
        pass

    @property
    @abstractmethod
    def solvers(self) -> dict[str, LinearSolver]:
        """Linear solvers keyed by field name from ``fvSolution``."""
        pass

    @abstractmethod
    def step(self):
        """Advance the simulation by one algorithm step."""
        pass

    def set_grad_mode(self, grad_mode: GradientMode) -> None:
        """
        Set ``grad_mode`` on every registered linear solver.

        Parameters
        ----------
        grad_mode : {"adjoint", "unrolled"}
            Differentiation mode applied to each solver.
        """
        for solver in self.solvers.values():
            solver.grad_mode = grad_mode

    def attach_diagnostics(self, diagnostics: DiagnosticsCollector) -> None:
        """
        Attach a diagnostics collector for per-step CSV recording.

        Parameters
        ----------
        diagnostics : DiagnosticsCollector
            Collector that receives continuity and solver statistics.
        """
        self._diagnostics = diagnostics

    def _finalize_diagnostics(
        self,
        phi: FaceField,
        solve_stats: dict[str, tuple[SolveStats, ...]],
    ) -> None:
        """
        Record enabled diagnostics for the completed algorithm step.

        Parameters
        ----------
        phi : FaceField
            Face flux field after the step completes.
        solve_stats : dict[str, tuple[SolveStats, ...]]
            Per-component solver statistics keyed by field name for the
            current step.
        """
        if self._diagnostics is None:
            return
        self._diagnostics_step += 1
        self._diagnostics.record_step(
            self._diagnostics_step,
            phi,
            solve_stats,
        )
