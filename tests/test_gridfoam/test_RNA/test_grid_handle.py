"""Tests for RNA/grid_handle module."""

import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from gridfoam.DNA._grid._grid import NodeType, PyBBox, PyCubeCode, PyGrid, PyOctreeLevel, PyOctreeNode
from gridfoam.DNA.config import CubeConfig, GridfoamConfig
from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.RNA.grid_handle import GridHandle


def test_grid_handle_import():
    """Test that GridHandle can be imported."""
    from gridfoam.RNA.grid_handle import GridHandle
    assert GridHandle is not None


def test_grid_handle_init_with_mock():
    """Test GridHandle initialization with mocked grid generation."""
    # Create minimal mock grid
    mock_bbox = PyBBox()
    mock_bbox.lower = np.array([0.0, 0.0, 0.0])
    mock_bbox.upper = np.array([1.0, 1.0, 1.0])
    
    mock_level = PyOctreeLevel()
    mock_level.depth = 0
    mock_level.bounds = np.array([1, 1, 1], dtype=np.uint64)
    mock_level.nodes = {}
    mock_level.n_leaf_nodes = 0
    
    mock_grid = PyGrid()
    mock_grid.domain = mock_bbox
    mock_grid.blocksize = np.array([1, 1, 1], dtype=np.uint64)
    mock_grid.max_depth = 0
    mock_grid.octree_levels = [mock_level]
    mock_grid.n_leaf_nodes = 0
    
    # Create minimal mock mesh
    import pyvista as pv
    mock_mesh = pv.PolyData()
    
    # Create minimal mock config
    mock_config = MagicMock()
    mock_config.cube.interior_width = 8
    mock_config.cube.halo_width = 2
    mock_config.cube.device = torch.device("cpu")
    
    # Patch generate_grid_from_polydata and file operations
    with patch("gridfoam.RNA.grid_handle.generate_grid_from_polydata", return_value=mock_grid), \
         patch("gridfoam.RNA.grid_handle.pv.read", return_value=mock_mesh), \
         patch("gridfoam.RNA.grid_handle.YamlRoot") as mock_yaml_root:
        
        # Mock YAML loading
        mock_yaml_config = MagicMock()
        mock_yaml_config.gridfoam = mock_config
        mock_yaml_root.model_validate.return_value = mock_yaml_config
        
        # Create temporary config file path
        configpath = pathlib.Path("dummy.yaml")
        
        # GridHandle should initialize without file dependency
        grid_handle = GridHandle(configpath=configpath)
        
        assert grid_handle.grid == mock_grid
        assert grid_handle.mesh == mock_mesh
        assert grid_handle.config == mock_config
        assert hasattr(grid_handle, "iter_levels")
        assert hasattr(grid_handle, "iter_all_leaves")
        assert hasattr(grid_handle, "allocate_field")
        assert hasattr(grid_handle, "sync_halo")


def test_grid_handle_iter_levels():
    """Test GridHandle iter_levels method."""
    mock_level = PyOctreeLevel()
    mock_level.depth = 0
    mock_level.bounds = np.array([1, 1, 1], dtype=np.uint64)
    mock_level.nodes = {}
    mock_level.n_leaf_nodes = 0
    
    mock_grid = PyGrid()
    mock_grid.octree_levels = [mock_level]
    
    mock_mesh = MagicMock()
    mock_config = MagicMock()
    mock_config.cube.interior_width = 8
    
    with patch("gridfoam.RNA.grid_handle.generate_grid_from_polydata", return_value=mock_grid), \
         patch("gridfoam.RNA.grid_handle.pv.read", return_value=mock_mesh), \
         patch("gridfoam.RNA.grid_handle.YamlRoot") as mock_yaml_root:
        mock_yaml_config = MagicMock()
        mock_yaml_config.gridfoam = mock_config
        mock_yaml_root.model_validate.return_value = mock_yaml_config
        
        configpath = pathlib.Path("dummy.yaml")
        grid_handle = GridHandle(configpath=configpath)
        
        levels = list(grid_handle.iter_levels())
        assert len(levels) == 1
        assert levels[0].depth == 0


def test_grid_handle_iter_all_leaves():
    """Test GridHandle iter_all_leaves method."""
    # Create mock leaf node
    mock_cube = PyOctreeNode()
    mock_cube.node_type = NodeType.LEAF
    mock_cube.cubecode = MagicMock()
    mock_cube.field = MagicMock()
    
    mock_level = PyOctreeLevel()
    mock_level.depth = 0
    mock_level.bounds = np.array([1, 1, 1], dtype=np.uint64)
    mock_level.nodes = {0: mock_cube}
    mock_level.n_leaf_nodes = 1
    
    mock_grid = PyGrid()
    mock_grid.octree_levels = [mock_level]
    
    mock_mesh = MagicMock()
    mock_config = MagicMock()
    mock_config.cube.interior_width = 8
    
    with patch("gridfoam.RNA.grid_handle.generate_grid_from_polydata", return_value=mock_grid), \
         patch("gridfoam.RNA.grid_handle.pv.read", return_value=mock_mesh), \
         patch("gridfoam.RNA.grid_handle.YamlRoot") as mock_yaml_root:
        mock_yaml_config = MagicMock()
        mock_yaml_config.gridfoam = mock_config
        mock_yaml_root.model_validate.return_value = mock_yaml_config
        
        configpath = pathlib.Path("dummy.yaml")
        grid_handle = GridHandle(configpath=configpath)
        
        leaves = list(grid_handle.iter_all_leaves())
        assert len(leaves) == 1
        depth, cube = leaves[0]
        assert depth == 0
        assert cube == mock_cube


def test_grid_handle_allocate_field():
    """Test GridHandle allocate_field method."""
    mock_cube = PyOctreeNode()
    mock_cube.node_type = NodeType.LEAF
    mock_cube.cubecode = MagicMock()
    from gridfoam.DNA.cubefield import CubeField
    mock_cube.field = CubeField(CubeConfig(interior_width=8, halo_width=2, device=torch.device("cpu")))
    
    mock_level = PyOctreeLevel()
    mock_level.depth = 0
    mock_level.bounds = np.array([1, 1, 1], dtype=np.uint64)
    mock_level.nodes = {0: mock_cube}
    mock_level.n_leaf_nodes = 1
    
    mock_grid = PyGrid()
    mock_grid.octree_levels = [mock_level]
    
    mock_mesh = MagicMock()
    mock_config = MagicMock()
    mock_config.cube.interior_width = 8
    mock_config.cube.halo_width = 2
    mock_config.cube.device = torch.device("cpu")
    
    with patch("gridfoam.RNA.grid_handle.generate_grid_from_polydata", return_value=mock_grid), \
         patch("gridfoam.RNA.grid_handle.pv.read", return_value=mock_mesh), \
         patch("gridfoam.RNA.grid_handle.YamlRoot") as mock_yaml_root:
        mock_yaml_config = MagicMock()
        mock_yaml_config.gridfoam = mock_config
        mock_yaml_root.model_validate.return_value = mock_yaml_config
        
        configpath = pathlib.Path("dummy.yaml")
        grid_handle = GridHandle(configpath=configpath)
        
        field_meta = FieldMeta(
            name="test_field",
            label="Test Field",
            layout=FieldLayout.CELL,
            role=FieldRole.STATE,
            components=1,
            dtype=torch.float32,
        )
        
        grid_handle.allocate_field(field_meta)
        
        # Verify field was allocated
        for _, cube in grid_handle.iter_all_leaves():
            assert field_meta.name in cube.field.cells
            break


def test_grid_handle_get_dx_at_depth():
    """Test GridHandle get_dx_at_depth method."""
    mock_bbox = PyBBox()
    mock_bbox.lower = np.array([0.0, 0.0, 0.0])
    mock_bbox.upper = np.array([2.0, 2.0, 2.0])
    
    mock_level = PyOctreeLevel()
    mock_level.depth = 0
    mock_level.bounds = np.array([2, 2, 2], dtype=np.uint64)
    mock_level.nodes = {}
    mock_level.n_leaf_nodes = 0
    
    mock_grid = PyGrid()
    mock_grid.domain = mock_bbox
    mock_grid.octree_levels = [mock_level]
    
    mock_mesh = MagicMock()
    mock_config = MagicMock()
    mock_config.cube.interior_width = 8
    
    with patch("gridfoam.RNA.grid_handle.generate_grid_from_polydata", return_value=mock_grid), \
         patch("gridfoam.RNA.grid_handle.pv.read", return_value=mock_mesh), \
         patch("gridfoam.RNA.grid_handle.YamlRoot") as mock_yaml_root:
        mock_yaml_config = MagicMock()
        mock_yaml_config.gridfoam = mock_config
        mock_yaml_root.model_validate.return_value = mock_yaml_config
        
        configpath = pathlib.Path("dummy.yaml")
        grid_handle = GridHandle(configpath=configpath)
        
        dx = grid_handle.get_dx_at_depth(0)
        assert dx.shape == (3,)
        # Expected: (2.0 - 0.0) / (2 * 8) = 0.125 for each direction
        expected_dx = torch.tensor([0.125, 0.125, 0.125])
        torch.testing.assert_close(dx, expected_dx, rtol=1e-6, atol=1e-6)
