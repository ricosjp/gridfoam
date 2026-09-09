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

    def validate(self, algorithm: AlgorithmBase) -> None:
        """
        Check instance and field compatibility without restoring state.

        Parameters
        ----------
        algorithm : AlgorithmBase
            Algorithm the checkpoint will be restored onto.

        Raises
        ------
        ValueError
            If ``algorithm`` is not the captured instance, or if
            :meth:`GridCheckpoint.validate` rejects its grid.
        """
        if algorithm is not self._algorithm:
            raise ValueError("Checkpoint belongs to another algorithm instance")
        self.fields.validate(algorithm.grid)

    def restore(self, algorithm: AlgorithmBase) -> None:
        """
        Restore field buffers, histories and iteration controls.

        Parameters
        ----------
        algorithm : AlgorithmBase
            Algorithm the checkpoint is restored onto.

        Raises
        ------
        ValueError
            If :meth:`validate` would reject ``algorithm``. Nothing is written
            in that case.
        """
        if algorithm is not self._algorithm:
            raise ValueError("Checkpoint belongs to another algorithm instance")
        # GridCheckpoint.restore validates the grid itself before writing.
        self.fields.restore(algorithm.grid)
        algorithm.restore_iteration_state(self.iteration)
