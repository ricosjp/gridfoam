import pathlib
import tempfile

import pytest
import pyvista as pv
import torch
import yaml

from gridfoam._base._tensor_grid import TensorGrid
from gridfoam.config import Config
from gridfoam.cubion import NodeType, PyGrid, RawIndexConversionMode
from gridfoam.utils.enums import Direction


class TestTensorGrid:
    """Test TensorGrid class"""

    @pytest.fixture
    def test_config_path(self) -> pathlib.Path:
        """Path to test configuration file"""
        return pathlib.Path("tests/data/yaml/bunny.yaml")

    @pytest.fixture
    def tensor_grid(self, test_config_path: pathlib.Path) -> TensorGrid:
        """TensorGrid fixture"""
        return TensorGrid.build(test_config_path)

    def test_tensor_grid_initialization(self, tensor_grid: TensorGrid):
        """Test TensorGrid initialization"""
        assert tensor_grid.data is not None
        assert tensor_grid.config is not None
        assert tensor_grid.field_dict == {}
        assert tensor_grid.mesh is not None
        assert tensor_grid.device == torch.device("cpu")

    def test_tensor_grid_build_from_config(
        self, test_config_path: pathlib.Path
    ):
        """Test TensorGrid.build method"""
        tensor_grid = TensorGrid.build(test_config_path)

        assert isinstance(tensor_grid.data, PyGrid)
        assert isinstance(tensor_grid.config, Config)
        assert isinstance(tensor_grid.mesh, pv.PolyData)
        assert tensor_grid.device == torch.device("cpu")
        assert tensor_grid.field_dict == {}

        domain_width = tensor_grid.domain_width
        assert isinstance(domain_width, torch.Tensor)
        assert domain_width.shape == (3,)
        assert domain_width.dtype == torch.float32
        assert domain_width.device == torch.device("cpu")

    def test_add_field(self, tensor_grid: TensorGrid):
        """Test add_field method"""
        # Add vector field
        tensor_grid.add_field("velocity", (3,), torch.float32)
        assert "velocity" in tensor_grid.field_dict
        assert tensor_grid.field_dict["velocity"] == ((3,), torch.float32)

        # Add scalar field
        tensor_grid.add_field("pressure", (1,), torch.float64)
        assert "pressure" in tensor_grid.field_dict
        assert tensor_grid.field_dict["pressure"] == ((1,), torch.float64)

    def test_add_field_invalid_shape(self, tensor_grid: TensorGrid):
        """Test add_field with invalid shape"""
        with pytest.raises(ValueError, match="All elements must be > 0"):
            tensor_grid.add_field("invalid", (0,), torch.float32)

        with pytest.raises(ValueError, match="All elements must be > 0"):
            tensor_grid.add_field("invalid", (-1, 2), torch.float32)

    def test_allocate_field_tensors(self, tensor_grid: TensorGrid):
        """Test allocate_field_tensors method"""
        # Add some fields
        tensor_grid.add_field("velocity", (3,), torch.float32)
        tensor_grid.add_field("pressure", (1,), torch.float64)

        # Allocate field tensors
        tensor_grid.allocate_field_tensors()

        # Check that field tensors are allocated for all nodes
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                assert "velocity" in node.field_tensors
                assert "pressure" in node.field_tensors

                # Check tensor properties
                velocity_tensor = node.field_tensors["velocity"]
                pressure_tensor = node.field_tensors["pressure"]

                assert velocity_tensor.width == tensor_grid.config.cube.width
                assert velocity_tensor.bnd == tensor_grid.config.cube.bnd_width
                assert velocity_tensor.raw.dtype == torch.float32
                assert velocity_tensor.raw.device == tensor_grid.device

                assert pressure_tensor.width == tensor_grid.config.cube.width
                assert pressure_tensor.bnd == tensor_grid.config.cube.bnd_width
                assert pressure_tensor.raw.dtype == torch.float64
                assert pressure_tensor.raw.device == tensor_grid.device

    def test_allocate_field_tensors_empty_field_dict(
        self, tensor_grid: TensorGrid
    ):
        """Test allocate_field_tensors with empty field_dict"""
        # Should not raise any errors
        tensor_grid.allocate_field_tensors()

        # Check that no field tensors are allocated
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                assert len(node.field_tensors) == 0

    def test_update_halo_with_field_tensors(self, tensor_grid: TensorGrid):
        """Test update_halo method with allocated field tensors"""
        # Add and allocate field tensors
        tensor_grid.add_field("velocity", (3,), torch.float32)
        tensor_grid.allocate_field_tensors()

        # Initialize some test data
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                # Set some test data in interior
                node.field_tensors["velocity"].interior.fill_(1.0)

        # Update halo - should not raise errors
        tensor_grid.update_halo()

        # Verify that halo update completed without errors
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                if node.node_type != NodeType.LEAF:
                    continue
                nbr_codes = node.cubecode.neighbor_codes(
                    octree_level.depth,
                    octree_level.bounds,
                    RawIndexConversionMode.BORDER,
                    False,
                )
                if nbr_codes[Direction.XM.value] is not None:
                    xm = torch.all(node.field_tensors["velocity"].xm == 1.0)
                    assert xm
                if nbr_codes[Direction.XP.value] is not None:
                    xp = torch.all(node.field_tensors["velocity"].xp == 1.0)
                    assert xp
                if nbr_codes[Direction.YM.value] is not None:
                    ym = torch.all(node.field_tensors["velocity"].ym == 1.0)
                    assert ym
                if nbr_codes[Direction.YP.value] is not None:
                    yp = torch.all(node.field_tensors["velocity"].yp == 1.0)
                    assert yp
                if nbr_codes[Direction.ZM.value] is not None:
                    zm = torch.all(node.field_tensors["velocity"].zm == 1.0)
                    assert zm
                if nbr_codes[Direction.ZP.value] is not None:
                    zp = torch.all(node.field_tensors["velocity"].zp == 1.0)
                    assert zp

    def test_update_halo_no_field_tensors(self, tensor_grid: TensorGrid):
        """Test update_halo method without field tensors"""
        # Should raise AttributeError when trying to access field_tensors
        with pytest.raises(AttributeError):
            tensor_grid.update_halo()

    def test_sync_ghost_from_parent_with_field_tensors(
        self, tensor_grid: TensorGrid
    ):
        """Test sync_ghost_from_parent method with allocated field tensors"""
        # Add and allocate field tensors
        tensor_grid.add_field("velocity", (3,), torch.float32)
        tensor_grid.allocate_field_tensors()

        # Initialize parent data
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                if node.node_type == NodeType.LEAF:
                    # Set some test data in parent
                    node.field_tensors["velocity"].interior.fill_(2.0)

        # Sync ghost from parent - should not raise errors
        tensor_grid.sync_ghost_from_parent()

        # Verify that sync completed without errors
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                if node.node_type == NodeType.GHOST_FROM_PARENT:
                    # Set some test data in parent
                    assert torch.all(node.field_tensors["velocity"].interior == 2.0)

    def test_sync_ghost_from_children_with_field_tensors(
        self, tensor_grid: TensorGrid
    ):
        """Test sync_ghost_from_children method with allocated field tensors"""
        # Add and allocate field tensors
        tensor_grid.add_field("velocity", (3,), torch.float32)
        tensor_grid.allocate_field_tensors()

        # Initialize child data
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                if node.node_type == NodeType.LEAF:
                    # Set some test data in children
                    node.field_tensors["velocity"].interior.fill_(3.0)

        # Sync ghost from children - should not raise errors
        tensor_grid.sync_ghost_from_children()

        # Verify that sync completed without errors
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                if node.node_type == NodeType.GHOST_FROM_CHILD:
                    # Set some test data in parent
                    assert torch.all(node.field_tensors["velocity"].interior == 3.0)

    def test_sync_ghost_methods_no_field_tensors(
        self, tensor_grid: TensorGrid
    ):
        """Test ghost sync methods without field tensors"""
        # Should raise AttributeError when trying to access field_tensors
        with pytest.raises(AttributeError):
            tensor_grid.sync_ghost_from_parent()

        with pytest.raises(AttributeError):
            tensor_grid.sync_ghost_from_children()

    @pytest.mark.with_device
    def test_tensor_grid_with_different_device(
        self, test_config_path: pathlib.Path
    ):
        """Test TensorGrid with different device (if CUDA available)"""
        with open(test_config_path) as f:
            config_dict = yaml.safe_load(f)

        config_dict["device"] = "cuda:0"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
            temp_config_path = pathlib.Path(f.name)
            try:
                tensor_grid = TensorGrid.build(temp_config_path)
                assert tensor_grid.device == torch.device("cuda:0")
            finally:
                temp_config_path.unlink()

    def test_field_tensor_shapes(self, tensor_grid: TensorGrid):
        """Test that field tensors have correct shapes"""
        # Add fields with different shapes
        tensor_grid.add_field("scalar", (1,), torch.float32)
        tensor_grid.add_field("vector", (3,), torch.float32)
        tensor_grid.add_field("tensor", (3, 3), torch.float32)

        tensor_grid.allocate_field_tensors()

        data_width = (
            tensor_grid.config.cube.width
            + 2 * tensor_grid.config.cube.bnd_width
        )

        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                # Check scalar field shape
                scalar_shape = node.field_tensors["scalar"].raw.shape
                expected_scalar_shape = (data_width, data_width, data_width, 1)
                assert scalar_shape == expected_scalar_shape

                # Check vector field shape
                vector_shape = node.field_tensors["vector"].raw.shape
                expected_vector_shape = (data_width, data_width, data_width, 3)
                assert vector_shape == expected_vector_shape

                # Check tensor field shape
                tensor_shape = node.field_tensors["tensor"].raw.shape
                expected_tensor_shape = (
                    data_width,
                    data_width,
                    data_width,
                    3,
                    3,
                )
                assert tensor_shape == expected_tensor_shape
