"""Named, graph-preserving tensor I/O for prepared grid fields."""

from collections.abc import Mapping
from types import MappingProxyType

import torch

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.shapes import require_shape
from gridfoam.core.state import TensorState


class FieldBindings:
    """
    Bind external tensor keys to existing cell or face fields on one grid.

    Strong references keep the weak field registry alive. Keys are independent
    of field names, but each field may occur only once. Tensors use gridfoam's
    physical axes, without a feature or batch axis. Feature conversion and
    physical-dimension checks at an application boundary belong to its adapter.

    Reads and writes copy storage while preserving autograd links. This is
    current-field I/O, not a complete algorithm state or a replay checkpoint.
    Prepare all fields before constructing checkpoints or freezing a registry.

    Parameters
    ----------
    fields : Mapping[str, CellField or FaceField]
        External key to registered field. All fields must belong to the same
        grid and each field object may appear once.

    Raises
    ------
    ValueError
        If ``fields`` is empty, a field appears under several keys, the fields
        belong to different grids, or a field is no longer the object
        registered under its name.
    """

    def __init__(self, fields: Mapping[str, CellField | FaceField]):
        if not fields:
            raise ValueError("Field bindings must be nonempty")
        if len({id(field) for field in fields.values()}) != len(fields):
            raise ValueError("A field cannot have multiple writable bindings")
        self._grid = next(iter(fields.values())).grid
        if any(field.grid is not self._grid for field in fields.values()):
            raise ValueError("Bound fields must belong to the same grid")
        self._fields = MappingProxyType(dict(fields))
        self._validate_registration()

    @property
    def fields(self) -> Mapping[str, CellField | FaceField]:
        """Read-only binding map; fields expose shape and dimension metadata."""
        return self._fields

    def _validate_registration(self) -> None:
        for key, field in self._fields.items():
            registered = (
                self._grid.get_cellfield(field.name)
                if isinstance(field, CellField)
                else self._grid.get_facefield(field.name)
            )
            if registered is not field:
                raise ValueError(f"Bound field was replaced: {key!r}")

    def validate(self, values: Mapping[str, torch.Tensor]) -> None:
        """
        Validate every key, shape, dtype and device without writing.

        Parameters
        ----------
        values : Mapping[str, torch.Tensor]
            Tensors keyed by binding key. Key order is irrelevant.

        Raises
        ------
        ValueError
            If the key set differs from the bindings, a tensor does not have
            the field's physical shape (``(C, *physical)`` for cells,
            ``(packed_n_rows, *physical)`` for faces), a tensor does not match
            the grid dtype or device, or a bound field has been replaced.
        """
        self._validate_registration()
        if values.keys() != self._fields.keys():
            raise ValueError("Field binding keys differ")
        for key, field in self._fields.items():
            value = values[key]
            rows = (
                self._grid.num_cells
                if isinstance(field, CellField)
                else field.packed_n_rows()
            )
            require_shape(value, (rows, *field.component_shape), key)
            if (
                value.dtype != self._grid.dtype
                or value.device != self._grid.device
            ):
                raise ValueError(f"Field {key!r} must match grid dtype/device")

    def read(self) -> TensorState:
        """
        Read current values with independent storage and live gradients.

        Returns
        -------
        TensorState
            Cloned cell data or packed face data keyed by binding key. Later
            writes to the fields do not modify the returned tensors.

        Raises
        ------
        ValueError
            If a bound field has been replaced in the grid registry.
        """
        self._validate_registration()
        values = TensorState(
            {
                key: field.data
                if isinstance(field, CellField)
                else field.pack()
                for key, field in self._fields.items()
            }
        )
        return values.clone()

    def write(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        reset_history: bool = False,
    ) -> None:
        """
        Replace current values after validating the entire input mapping.

        Tensors are cloned so the caller's leaves are never mutated by later
        in-place solver updates. No dtype/device conversion, broadcasting or
        detachment is performed. Field-level FV caches are keyed by the data
        tensor and invalidate themselves; no grid-level cache is cleared.

        Parameters
        ----------
        values : Mapping[str, torch.Tensor]
            Tensors keyed by binding key; see :meth:`validate` for the
            accepted layout.
        reset_history : bool, default False
            If False, time histories are untouched. If True, each written
            field starts a new initial state and discards older levels. Do not
            use this to restore an established transient trajectory; use a
            checkpoint for that.

        Raises
        ------
        ValueError
            If :meth:`validate` rejects ``values``. Nothing is written in that
            case.
        """
        self.validate(values)
        copies = TensorState(values).clone()
        for key, field in self._fields.items():
            if isinstance(field, CellField):
                field.data = copies[key]
            else:
                field.replace_packed(copies[key])
            if reset_history:
                field.update_history(reset=True)
