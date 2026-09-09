"""Field checkpoints for replay on a fixed grid and configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.base import GridBase
from gridfoam.core.state import TensorState, validate_history


def _registered_fields(grid: GridBase) -> dict[str, CellField | FaceField]:
    fields: dict[str, CellField | FaceField] = {}
    for name in sorted(grid.cellfield_names()):
        cell = grid.get_cellfield(name)
        if cell is not None:
            fields[f"cell/{name}"] = cell
    for name in sorted(grid.facefield_names()):
        face = grid.get_facefield(name)
        if face is not None:
            fields[f"face/{name}"] = face
    return fields


@dataclass(frozen=True)
class GridCheckpoint:
    """All registered fields, including scratch fields and time histories.

    This is an in-memory replay checkpoint, not a portable case file or the
    minimal fixed-point unknown vector. Boundary/model parameters are inputs,
    not checkpointed state: replay callers must reapply them explicitly.
    Grid/configuration changes require a new checkpoint, including equal-size
    remeshing and IBM updates. Field references keep the weak registry alive.
    """

    values: TensorState
    previous_dts: Mapping[str, float | None]
    aliased_old: frozenset[str]
    _fields: Mapping[str, CellField | FaceField]
    _grid: GridBase
    _config: str
    _revision: tuple[int, int]
    _dt: float

    @classmethod
    def capture(cls, grid: GridBase, *, detach: bool = True) -> GridCheckpoint:
        """Copy registered buffers; optionally retain their autograd links."""
        fields = _registered_fields(grid)
        values: dict[str, torch.Tensor] = {}
        previous_dts: dict[str, float | None] = {}
        aliased_old: set[str] = set()
        for key, field in fields.items():
            if isinstance(field, CellField):
                data, old, older = field.data, field.old_data, field.older_data
                if data is old:
                    aliased_old.add(key)
            else:
                data = field.pack()
                old, older = field.old_single_data, field.older_single_data
            values[f"{key}/data"] = data
            if key not in aliased_old:
                values[f"{key}/old"] = old
            if older is not None:
                values[f"{key}/older"] = older
            previous_dts[key] = field.previous_dt
        blocks = TensorState(values)
        return cls(
            values=blocks.checkpoint() if detach else blocks.clone(),
            previous_dts=MappingProxyType(previous_dts),
            aliased_old=frozenset(aliased_old),
            _fields=MappingProxyType(fields),
            _grid=grid,
            # Frozen Pydantic models still contain mutable dictionaries.
            _config=grid.sim_config.model_dump_json(),
            _revision=(grid.topology_revision, grid.geometry_revision),
            _dt=grid.dt,
        )

    def with_values(self, values: TensorState) -> GridCheckpoint:
        """Replace differentiable blocks without changing replay metadata."""
        self.values.validate_layout(values)
        return replace(self, values=values)

    def validate(self, grid: GridBase) -> None:
        """
        Check replay compatibility without modifying fields or caches.

        Boundary values, solver settings and model parameters remain the
        caller's fixed-input contract; they are not captured or compared here.

        Parameters
        ----------
        grid : GridBase
            Grid the checkpoint will be restored onto.

        Raises
        ------
        ValueError
            If ``grid`` is a different object, its geometry or topology
            generation, configuration or time step changed, the set or
            identity of registered fields changed, or a buffer layout differs.
        """
        if (
            grid is not self._grid
            or (grid.topology_revision, grid.geometry_revision)
            != self._revision
        ):
            raise ValueError("Checkpoint grid or geometry has changed")
        if (
            grid.sim_config.model_dump_json() != self._config
            or grid.dt != self._dt
        ):
            raise ValueError(
                "Checkpoint configuration or time step has changed"
            )
        current_fields = _registered_fields(grid)
        if current_fields.keys() != self._fields.keys() or any(
            current_fields[key] is not field
            for key, field in self._fields.items()
        ):
            raise ValueError("Registered checkpoint fields have changed")

        for key, field in self._fields.items():
            data = self.values[f"{key}/data"]
            reference = (
                field.data if isinstance(field, CellField) else field.pack()
            )
            if (
                data.shape != reference.shape
                or data.dtype != reference.dtype
                or data.device != reference.device
            ):
                raise ValueError(f"Checkpoint layout differs for {key!r}")
            history_reference = (
                data
                if isinstance(field, CellField)
                else data[: field.num_single_sided]
            )
            old = data if key in self.aliased_old else self.values[f"{key}/old"]
            validate_history(
                history_reference,
                old,
                self.values.get(f"{key}/older"),
                self.previous_dts[key],
            )

    def restore(self, grid: GridBase) -> None:
        """
        Restore independent buffers, preserving the checkpoint's graph.

        Every field is validated before any buffer is written. Restoration
        does not call ``update_history`` and clears all derived FV caches so a
        no-grad or already-consumed graph cannot leak into a new replay.

        Parameters
        ----------
        grid : GridBase
            Grid the checkpoint is restored onto.

        Raises
        ------
        ValueError
            If :meth:`validate` rejects ``grid``. Nothing is written in that
            case.
        """
        self.validate(grid)
        # Clone even when the source has no graph: an in-place solver update
        # must never mutate the checkpoint used by a later replay.
        values = self.values.clone()
        for key, field in self._fields.items():
            data = values[f"{key}/data"]
            if isinstance(field, CellField):
                field.data = data
            else:
                field.replace_packed(data)
            old = data if key in self.aliased_old else values[f"{key}/old"]
            field.restore_history(
                old, values.get(f"{key}/older"), self.previous_dts[key]
            )
        grid.invalidate_derived_caches()
