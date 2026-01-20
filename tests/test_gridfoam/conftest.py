"""Pytest configuration and fixtures for gridfoam tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from gridfoam.DNA._grid._grid import PyOctreeNode
from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta


@pytest.fixture
def cube_config() -> CubeConfig:
    """Create a test CubeConfig."""
    return CubeConfig(
        interior_width=8,
        halo_width=2,
        device=torch.device("cpu"),
    )


@pytest.fixture
def cube_field(cube_config: CubeConfig) -> CubeField:
    """Create a test CubeField."""
    return CubeField(cube_config)


@pytest.fixture
def mock_cube_node(cube_field: CubeField) -> PyOctreeNode:
    """Create a mock PyOctreeNode with a CubeField."""
    mock_node = MagicMock(spec=PyOctreeNode)
    mock_node.field = cube_field
    # Mock cubecode.neighbor_codes to return all None (no neighbors)
    mock_node.cubecode.neighbor_codes = MagicMock(return_value=[None] * 27)
    return mock_node


@pytest.fixture
def ctx_for_cube_operation() -> CtxForCubeOperation:
    """Create a test CtxForCubeOperation."""
    return CtxForCubeOperation(
        depth=0,
        bounds=np.array([1, 1, 1], dtype=np.uint64),
        dt=0.1,
        dx=torch.tensor([1.0, 1.0, 1.0]),
        vertices=torch.zeros((8, 3)),
        bcs=[],
    )


@pytest.fixture
def cell_field_meta() -> FieldMeta:
    """Create a test cell-centered FieldMeta."""
    return FieldMeta(
        name="test_cell",
        label="Test Cell",
        layout=FieldLayout.CELL,
        role=FieldRole.STATE,
        components=1,
        dtype=torch.float32,
    )


@pytest.fixture
def face_field_meta() -> FieldMeta:
    """Create a test face-centered FieldMeta."""
    return FieldMeta(
        name="test_face",
        label="Test Face",
        layout=FieldLayout.FACE,
        role=FieldRole.AUXILIARY,
        components=1,
        dtype=torch.float32,
    )
