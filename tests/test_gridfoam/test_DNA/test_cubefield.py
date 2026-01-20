"""Tests for cubefield module."""

import pytest
import torch

from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta


def test_cube_field_import():
    """Test that CubeField can be imported."""
    from gridfoam.DNA.cubefield import CubeField

    assert CubeField is not None


def test_cube_field_init():
    """Test CubeField initialization."""
    cube_config = CubeConfig(
        interior_width=8,
        halo_width=2,
        device=torch.device("cpu"),
    )
    cube_field = CubeField(cube_config)
    assert cube_field.N == 8
    assert cube_field.H == 2
    assert cube_field.device == torch.device("cpu")
    assert isinstance(cube_field.cells, dict)
    assert isinstance(cube_field.faces, dict)
    assert isinstance(cube_field.fvmatrices, dict)


def test_cube_field_add_cell_tensor():
    """Test CubeField _add_cell_tensor method."""
    cube_config = CubeConfig(
        interior_width=8,
        halo_width=2,
        device=torch.device("cpu"),
    )
    cube_field = CubeField(cube_config)
    field_meta = FieldMeta(
        name="test",
        label="Test",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
        components=1,
        dtype=torch.float32,
    )
    cube_field._add_cell_tensor(field_meta)
    assert "test" in cube_field.cells
    assert cube_field.cells["test"].N == 8
    assert cube_field.cells["test"].H == 2


def test_cube_field_add_face_tensor():
    """Test CubeField _add_face_tensor method."""
    cube_config = CubeConfig(
        interior_width=8,
        halo_width=2,
        device=torch.device("cpu"),
    )
    cube_field = CubeField(cube_config)
    field_meta = FieldMeta(
        name="test_face",
        label="Test Face",
        layout=FieldLayout.FACE,
        role=FieldRole.AUXILIARY,
        components=1,
        dtype=torch.float32,
    )
    cube_field._add_face_tensor(field_meta)
    assert "test_face" in cube_field.faces
    assert cube_field.faces["test_face"].N == 8
