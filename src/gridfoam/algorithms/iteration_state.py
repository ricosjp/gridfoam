"""Non-tensor iteration state needed to reproduce convergence decisions."""

from collections.abc import Mapping
from types import MappingProxyType


class IterationState:
    """Immutable snapshot, isolated from the caller's residual dictionaries.

    Residual mappings are copied at construction and exposed as read-only
    views. Restore writes independent mutable dictionaries back onto the
    algorithm.
    """

    def __init__(
        self,
        diagnostics_step: int,
        initial_residuals: Mapping[str, float] | None = None,
        current_residuals: Mapping[str, float] | None = None,
        outer_converged: bool = False,
    ) -> None:
        self._diagnostics_step = diagnostics_step
        self._initial_residuals: Mapping[str, float] = MappingProxyType(
            dict(initial_residuals) if initial_residuals is not None else {}
        )
        self._current_residuals: Mapping[str, float] = MappingProxyType(
            dict(current_residuals) if current_residuals is not None else {}
        )
        self._outer_converged = outer_converged

    @property
    def diagnostics_step(self) -> int:
        """Diagnostic record counter at the captured step."""
        return self._diagnostics_step

    @property
    def initial_residuals(self) -> Mapping[str, float]:
        """Copied residual magnitudes at the start of the captured iteration."""
        return self._initial_residuals

    @property
    def current_residuals(self) -> Mapping[str, float]:
        """Copied residual magnitudes at the end of the captured iteration."""
        return self._current_residuals

    @property
    def outer_converged(self) -> bool:
        """Whether the outer loop had already reported convergence."""
        return self._outer_converged

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IterationState):
            return NotImplemented
        return (
            self._diagnostics_step == other._diagnostics_step
            and self._initial_residuals == other._initial_residuals
            and self._current_residuals == other._current_residuals
            and self._outer_converged == other._outer_converged
        )

    def __repr__(self) -> str:
        return (
            "IterationState("
            f"diagnostics_step={self._diagnostics_step!r}, "
            f"initial_residuals={dict(self._initial_residuals)!r}, "
            f"current_residuals={dict(self._current_residuals)!r}, "
            f"outer_converged={self._outer_converged!r})"
        )
