"""Replay support shared by segregated algorithms."""

from __future__ import annotations

from dataclasses import dataclass

from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.iteration_state import IterationState
from gridfoam.core.checkpoint import GridCheckpoint


@dataclass(frozen=True)
class AlgorithmCheckpoint:
    """Field histories and iteration controls for one algorithm instance.

    Boundary conditions, model parameters and solver settings must be reapplied
    or held fixed by the caller. Diagnostic files are not rolled back. Future
    models with non-field evolving state must add explicit checkpoint support.
    """

    fields: GridCheckpoint
    iteration: IterationState
    _algorithm: AlgorithmBase

    @classmethod
    def capture(
        cls, algorithm: AlgorithmBase, *, detach: bool = True
    ) -> AlgorithmCheckpoint:
        return cls(
            GridCheckpoint.capture(algorithm.grid, detach=detach),
            algorithm.capture_iteration_state(),
            algorithm,
        )

    def restore(self, algorithm: AlgorithmBase) -> None:
        if algorithm is not self._algorithm:
            raise ValueError("Checkpoint belongs to another algorithm instance")
        self.fields.restore(algorithm.grid)
        algorithm.restore_iteration_state(self.iteration)
