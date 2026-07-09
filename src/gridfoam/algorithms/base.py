from abc import ABC, abstractmethod

from gridfoam.core.field import FaceField
from gridfoam.core.grid.base import IGridBase
from gridfoam.models.turbulence.base import TurbulenceModel
from gridfoam.post.diagnostics import DiagnosticsCollector
from gridfoam.solvers.base import SolveStats


class AlgorithmBase(ABC):
    """
    Abstract base class for CFD algorithms (macro solvers).
    """

    _diagnostics: DiagnosticsCollector | None = None
    _diagnostics_step: int = 0

    @property
    @abstractmethod
    def grid(self) -> IGridBase:
        """
        Return the computational grid.
        """
        pass

    @property
    @abstractmethod
    def turbulence(self) -> TurbulenceModel:
        """
        Return the turbulence model.
        """
        pass

    @abstractmethod
    def step(self):
        """
        Advance the simulation by one algorithm step.
        """
        pass

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
