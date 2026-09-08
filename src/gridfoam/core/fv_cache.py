"""Finite-volume caches owned by grids and cell fields.

Lives in ``core`` so :class:`~gridfoam.core.grid.base.GridBase` and
:class:`~gridfoam.core.field.CellField` can expose ``fv_cache`` without
importing ``fv``. Slots are filled lazily by FV helpers
(:func:`~gridfoam.fv.kernels.face_geometry.face_geometry`,
``fv.boundary_ops``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FvGridCache:
    """
    Grid-owned FV derived data.

    Attributes
    ----------
    face_geometry :
        Cached :class:`~gridfoam.fv.kernels.face_geometry.FaceGeometry`,
        or ``None`` until first use / after :meth:`clear`.
    boundary_masks :
        Read-only patch masks and their CPU face counts.
    """

    face_geometry: Any | None = None
    boundary_masks: dict[Any, Any] = field(default_factory=dict)

    def clear(self) -> None:
        """Drop all cached grid-level FV data."""
        self.face_geometry = None
        self.boundary_masks.clear()


@dataclass
class FvFieldCache:
    """
    Field-owned FV derived data (boundary batches and states).

    Attributes
    ----------
    boundary_batches_key, boundary_batches :
        Cached boundary-face batches and their invalidation key.
    boundary_states_key, boundary_states :
        Cached evaluated boundary states and their invalidation key.
    immersed_constraints_key, immersed_constraints :
        Fixed Dirichlet cell/face selections, without prescribed values
        or their autograd graph. Invalidated with boundary batches.
    """

    boundary_batches_key: Any | None = None
    boundary_batches: Any | None = None
    boundary_states_key: Any | None = None
    boundary_states: Any | None = None
    immersed_constraints_key: Any | None = None
    immersed_constraints: Any | None = None

    def clear(self) -> None:
        """Drop all cached field-level FV data."""
        self.boundary_batches_key = None
        self.boundary_batches = None
        self.boundary_states_key = None
        self.boundary_states = None
        self.immersed_constraints_key = None
        self.immersed_constraints = None

    def clear_boundary_batches(self) -> None:
        """Drop boundary batches and dependent states."""
        self.clear()

    def clear_boundary_states(self) -> None:
        """Drop evaluated boundary-state cache only."""
        self.boundary_states_key = None
        self.boundary_states = None
