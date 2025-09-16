import pathlib
import tempfile

import pytest
import torch

from gridfoam._base._tensor_grid import TensorGrid
from gridfoam._io._vtkhdf import save_grid
from gridfoam.config import Config
from gridfoam.cubion import NodeType


class TestSaveGrid:
    """Test save_grid function"""

    @pytest.fixture
    def test_config_path(self) -> pathlib.Path:
        """Path to test configuration file"""
        return pathlib.Path("tests/data/yaml/bunny.yaml")

    @pytest.fixture
    def tensor_grid(self, test_config_path: pathlib.Path) -> TensorGrid:
        """TensorGrid fixture"""
        return TensorGrid.build(test_config_path)

    @pytest.fixture
    def temp_output_dir(self) -> pathlib.Path:
        """Temporary output directory for testing"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield pathlib.Path(tmp_dir)

    def test_save_grid_basic(
        self, tensor_grid: TensorGrid, temp_output_dir: pathlib.Path
    ):
        """Test basic save_grid functionality"""
        # Modify config to use temp directory
        config_dict = tensor_grid.config.model_dump()
        config_dict["io"]["output_dir"] = str(temp_output_dir)
        config_dict["io"]["mode"] = "cube"
        config_dict["io"]["only_leaves"] = True
        config_dict["io"]["overwrite_file"] = True
        config = Config.model_validate(config_dict)
        tensor_grid.config = config

        # Save grid
        save_grid(tensor_grid)

        # Check that file was created
        output_file = temp_output_dir / "grid.vtkhdf"
        assert output_file.exists()

    def test_save_grid_file_exists_error(
        self, tensor_grid: TensorGrid, temp_output_dir: pathlib.Path
    ):
        """Test save_grid raises error when file exists and overwrite_file is False"""
        # Modify config to use temp directory
        config_dict = tensor_grid.config.model_dump()
        config_dict["io"]["output_dir"] = str(temp_output_dir)
        config_dict["io"]["mode"] = "cube"
        config_dict["io"]["only_leaves"] = True
        config_dict["io"]["overwrite_file"] = False
        config = Config.model_validate(config_dict)
        tensor_grid.config = config

        # Save grid first time
        save_grid(tensor_grid)

        # Try to save again - should raise error
        with pytest.raises(FileExistsError, match="File .* already exists"):
            save_grid(tensor_grid)

    def test_save_grid_cube_mode(
        self, tensor_grid: TensorGrid, temp_output_dir: pathlib.Path
    ):
        """Test save_grid with CUBE mode"""
        # Modify config to use temp directory and CUBE mode
        config_dict = tensor_grid.config.model_dump()
        config_dict["io"]["output_dir"] = str(temp_output_dir)
        config_dict["io"]["mode"] = "cube"
        config_dict["io"]["only_leaves"] = True
        config_dict["io"]["overwrite_file"] = True
        config = Config.model_validate(config_dict)
        tensor_grid.config = config

        # Save grid
        save_grid(tensor_grid)

        # Check that file was created
        output_file = temp_output_dir / "grid.vtkhdf"
        assert output_file.exists()

    def test_save_grid_only_leaves_false(
        self, tensor_grid: TensorGrid, temp_output_dir: pathlib.Path
    ):
        """Test save_grid with only_leaves=False"""
        # Modify config to use temp directory and only_leaves=False
        config_dict = tensor_grid.config.model_dump()
        config_dict["io"]["output_dir"] = str(temp_output_dir)
        config_dict["io"]["mode"] = "cube"
        config_dict["io"]["only_leaves"] = False
        config_dict["io"]["overwrite_file"] = True
        config = Config.model_validate(config_dict)
        tensor_grid.config = config

        # Save grid
        save_grid(tensor_grid)

        # Check that file was created
        output_file = temp_output_dir / "grid.vtkhdf"
        assert output_file.exists()

    def test_save_grid_with_field_data(
        self, tensor_grid: TensorGrid, temp_output_dir: pathlib.Path
    ):
        """Test save_grid with field data in CELL mode"""
        # Modify config to use temp directory and CELL mode
        config_dict = tensor_grid.config.model_dump()
        config_dict["io"]["output_dir"] = str(temp_output_dir)
        config_dict["io"]["mode"] = "cell"
        config_dict["io"]["only_leaves"] = True
        config_dict["io"]["overwrite_file"] = True
        config = Config.model_validate(config_dict)
        tensor_grid.config = config

        # Add field data
        tensor_grid.add_field("velocity", (3,), torch.float32)
        tensor_grid.add_field("pressure", (1,), torch.float64)
        tensor_grid.allocate_field_tensors()

        # Initialize some test data
        for octree_level in tensor_grid.data.octree_levels:
            for node in octree_level.nodes.values():
                if node.node_type == NodeType.LEAF:
                    # Set test data
                    node.field_tensors["velocity"].interior.fill_(1.0)
                    node.field_tensors["pressure"].interior.fill_(2.0)

        # Save grid
        save_grid(tensor_grid)

        # Check that file was created
        output_file = temp_output_dir / "grid.vtkhdf"
        assert output_file.exists()
