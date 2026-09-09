"""External field I/O preserves graphs and rejects invalid writes atomically."""

import gc
import weakref
from pathlib import Path

import pytest
import torch
from tests.helpers import channel_config

from gridfoam.core.checkpoint import GridCheckpoint
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.field_bindings import FieldBindings
from gridfoam.core.grid.base import GridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.fv import fvc
from gridfoam.meta.enums import FieldRole


@pytest.fixture
def grid(tmp_path: Path) -> GridBase:
    """Create an isolated grid for each field-registry test."""
    return create_grid(channel_config(tmp_path))


@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
def test_round_trip_preserves_physical_axes_and_gradients(
    grid: GridBase, shape: tuple[int, ...]
) -> None:
    """Round trips preserve physical axes and independent autograd graphs."""
    cell = CellField(grid, "input", FieldRole.LOCAL, shape)
    face = FaceField(grid, "input_face", FieldRole.LOCAL, shape)
    bindings = FieldBindings({"cells": cell, "faces": face})
    x = torch.randn_like(cell.data, requires_grad=True)
    y = torch.randn_like(face.pack(), requires_grad=True)
    bindings.write({"cells": x, "faces": y})
    first = bindings.read()
    first_loss = first["cells"].square().sum() + first["faces"].square().sum()

    # A second forward and in-place writes into its buffers must not damage
    # the first graph, the caller's inputs, or the independent read snapshot.
    # Writes are keyed, so the mapping order is irrelevant.
    bindings.write({"faces": y * 3, "cells": x * 3})
    second = bindings.read()
    second_loss = second["cells"].sum() + second["faces"].sum()
    with torch.no_grad():
        cell.data.zero_()
        face.single_data.zero_()
    torch.testing.assert_close(first["cells"], x)
    torch.testing.assert_close(first["faces"], y)
    first_grad = torch.autograd.grad(first_loss, (x, y))
    second_grad = torch.autograd.grad(second_loss, (x, y))
    torch.testing.assert_close(first_grad[0], 2 * x)
    torch.testing.assert_close(first_grad[1], 2 * y)
    torch.testing.assert_close(second_grad[0], torch.full_like(x, 3))
    torch.testing.assert_close(second_grad[1], torch.full_like(y, 3))


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "keys"])
def test_invalid_second_input_does_not_write_first(
    grid: GridBase, invalid: str
) -> None:
    """Invalid mappings are rejected before any bound field is changed."""
    cell = CellField(grid, "input", FieldRole.LOCAL, ())
    face = FaceField(grid, "input_face", FieldRole.LOCAL, ())
    bindings = FieldBindings({"cells": cell, "faces": face})
    before = bindings.read()
    values = {"cells": cell.data + 10, "faces": face.pack()}
    if invalid == "shape":
        values["faces"] = values["faces"].unsqueeze(-1)
    elif invalid == "dtype":
        values["faces"] = values["faces"].float()
    elif invalid == "device":
        values["faces"] = values["faces"].to("meta")
    else:
        values["extra"] = values.pop("faces")
    with pytest.raises(ValueError):
        bindings.write(values)
    after = bindings.read()
    for key in before:
        torch.testing.assert_close(after[key], before[key])


def test_write_keeps_grid_level_caches(grid: GridBase) -> None:
    """Value writes rely on data-keyed field caches, not grid cache clears."""
    cell = CellField(grid, "input", FieldRole.LOCAL, ())
    bindings = FieldBindings({"cells": cell})
    marker = object()
    grid.fv_cache.face_geometry = marker
    bindings.write({"cells": cell.data + 1})
    assert grid.fv_cache.face_geometry is marker


def test_bindings_keep_fields_alive_and_reject_replacement(
    grid: GridBase, tmp_path: Path
) -> None:
    """Bindings own their fields and reject ambiguous or stale mappings."""
    field = CellField(grid, "input", FieldRole.LOCAL, ())
    ref = weakref.ref(field)
    bindings = FieldBindings({"input": field})
    with pytest.raises(ValueError, match="multiple"):
        FieldBindings({"a": field, "b": field})
    other_grid = create_grid(channel_config(tmp_path))
    other = CellField(other_grid, "other", FieldRole.LOCAL, ())
    with pytest.raises(ValueError, match="same grid"):
        FieldBindings({"a": field, "b": other})
    del field
    gc.collect()
    assert ref() is grid.get_cellfield("input")
    assert ref() is not None
    replacement = CellField(grid, "input", FieldRole.LOCAL, ())
    with pytest.raises(ValueError, match="replaced"):
        bindings.read()
    assert grid.get_cellfield("input") is replacement


def test_history_reset_is_explicit(grid: GridBase) -> None:
    """Writes preserve history unless a new initial state is requested."""
    cell = CellField(grid, "transient", FieldRole.TRANSIENT, ())
    face = FaceField(grid, "transient_face", FieldRole.TRANSIENT, ())
    cell.update_history()
    face.update_history()
    cell_old, cell_older = cell.old_data, cell.older_data
    face_old, face_older = face.old_single_data, face.older_single_data
    bindings = FieldBindings({"cell": cell, "face": face})
    x = torch.ones_like(cell.data, requires_grad=True)
    y = torch.ones_like(face.pack(), requires_grad=True)
    bindings.write({"cell": x, "face": y})
    assert cell.old_data is cell_old and cell.older_data is cell_older
    assert face.old_single_data is face_old
    assert face.older_single_data is face_older
    bindings.write({"cell": x, "face": y}, reset_history=True)
    assert cell.older_data is None and cell.previous_dt is None
    assert face.older_single_data is None and face.previous_dt is None
    torch.testing.assert_close(cell.old_data, x)
    torch.testing.assert_close(face.old_single_data, y[: face.num_single_sided])
    gradients = torch.autograd.grad(
        cell.old_data.sum() + face.old_single_data.sum(), (x, y)
    )
    torch.testing.assert_close(gradients[0], torch.ones_like(x))
    expected_face = torch.zeros_like(y)
    expected_face[: face.num_single_sided] = 1
    torch.testing.assert_close(gradients[1], expected_face)


def test_registry_scope_pins_registered_fields(grid: GridBase) -> None:
    """A frozen weak registry keeps its registered fields alive."""
    field = CellField(grid, "prepared", FieldRole.LOCAL, ())
    ref = weakref.ref(field)
    with grid.freeze_field_registry():
        del field
        gc.collect()
        assert ref() is not None
    gc.collect()
    assert ref() is None


def test_registry_scope_rejects_new_and_replacement_fields(
    grid: GridBase,
) -> None:
    """A frozen registry allows only re-registration of the same object."""
    field = CellField(grid, "prepared", FieldRole.LOCAL, ())
    with grid.freeze_field_registry():
        grid.register_cellfield(field)
        with pytest.raises(ValueError, match="frozen"):
            CellField(grid, "prepared", FieldRole.LOCAL, ())
        with pytest.raises(ValueError, match="frozen"):
            CellField(grid, "new_cell", FieldRole.LOCAL, ())
        with pytest.raises(ValueError, match="frozen"):
            FaceField(grid, "new_face", FieldRole.LOCAL, ())


def test_registry_scope_unfreezes_after_nested_failure(grid: GridBase) -> None:
    """Nested scopes release the registry guard when an exception escapes."""
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with grid.freeze_field_registry(), grid.freeze_field_registry():
            raise RuntimeError("evaluation failed")
    new = CellField(grid, "new_cell", FieldRole.LOCAL, ())
    assert grid.get_cellfield("new_cell") is new


def test_prepared_fvc_outputs_survive_separate_backwards(
    grid: GridBase,
) -> None:
    """Prepared FVC fields support pending graphs and inference evaluation."""
    field = CellField(grid, "psi", FieldRole.LOCAL, ())
    gradient = CellField(grid, "grad(psi)", FieldRole.LOCAL, (3,))
    inputs = FieldBindings({"input": field})
    outputs = FieldBindings({"gradient": gradient})
    checkpoint = GridCheckpoint.capture(grid)
    x = torch.randn_like(field.data, requires_grad=True)
    y = torch.randn_like(field.data, requires_grad=True)

    def evaluate(value: torch.Tensor) -> torch.Tensor:
        with grid.freeze_field_registry():
            checkpoint.validate(grid)
            try:
                inputs.write({"input": value})
                assert fvc.grad(field) is gradient
                return outputs.read()["gradient"].square().sum()
            finally:
                checkpoint.restore(grid)

    first, second = evaluate(x), evaluate(y)
    # Trainer evaluation can surround a module with inference_mode. The
    # adapter must run FV caches with ordinary, versioned tensors throughout.
    with torch.inference_mode():
        inference_input = x.clone()
        with torch.inference_mode(False), torch.no_grad():
            evaluated = evaluate(inference_input)
    torch.testing.assert_close(evaluated, first)
    (gx,) = torch.autograd.grad(first, x)
    (gy,) = torch.autograd.grad(second, y)
    direction = torch.randn_like(x)
    eps = 1e-5
    for value, grad in ((x, gx), (y, gy)):
        with torch.no_grad():
            fd = (
                evaluate(value + eps * direction)
                - evaluate(value - eps * direction)
            ) / (2 * eps)
        torch.testing.assert_close(
            (grad * direction).sum(), fd, rtol=1e-7, atol=1e-7
        )


def test_checkpoint_validation_is_read_only_and_detects_changes(
    grid: GridBase,
) -> None:
    """Validation detects stale geometry without modifying fields or caches."""
    field = CellField(grid, "psi", FieldRole.LOCAL, ())
    checkpoint = GridCheckpoint.capture(grid)
    field.data = field.data + 3
    current = field.data
    marker = object()
    grid.fv_cache.face_geometry = marker
    checkpoint.validate(grid)
    assert field.data is current
    assert grid.fv_cache.face_geometry is marker
    grid.mark_geometry_changed()
    with pytest.raises(ValueError, match="geometry"):
        checkpoint.validate(grid)
    assert field.data is current
