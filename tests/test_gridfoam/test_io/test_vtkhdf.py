import pathlib
import tempfile

import pytest
import torch
import yaml

from gridfoam import TensorGrid
from gridfoam._io._vtkhdf import save_grid
from gridfoam.config import IoConfig
from gridfoam.utils.enums import GridMode


@pytest.fixture
def config_path():
    """Create a config."""
    config_path = pathlib.Path("tests/data/yaml/Debug.yaml")
    return config_path


@pytest.fixture
def grid(config_path: pathlib.Path):
    """Create a test grid with cell fields."""
    grid = TensorGrid.build(config_path)
    grid.add_cell_field("U", (3,), torch.float32)
    grid.add_cell_field("T", (1,), torch.float32)
    grid.allocate_field_tensors()
    return grid


@pytest.fixture
def io_config(config_path: pathlib.Path) -> dict:
    """Modify the config."""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return config_dict["io"]


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    return pathlib.Path(temp_dir)


class TestSaveGrid:
    """Test save_grid function."""

    def test_save_grid_cell_mode(
        self, grid: TensorGrid, temp_output_dir: pathlib.Path, io_config: dict
    ) -> None:
        """Test save_grid in CELL mode with field data."""
        # Update config for CELL mode
        io_config["output_dir"] = temp_output_dir
        io_config["mode"] = GridMode.CELL
        io_config["only_leaves"] = True
        io_config["overwrite_file"] = True
        grid.config = grid.config.model_copy(
            update={"io": IoConfig.model_validate(io_config)}
        )

        # Save the grid
        save_grid(grid, "test_cell_mode.vtkhdf")

        # Verify file and field data
        output_file = temp_output_dir / "test_cell_mode.vtkhdf"
        assert output_file.exists()

    def test_save_grid_cube_mode(
        self, grid: TensorGrid, temp_output_dir: pathlib.Path, io_config: dict
    ) -> None:
        """Test save_grid in CUBE mode without field data."""
        # Update config for CUBE mode
        io_config["output_dir"] = temp_output_dir
        io_config["mode"] = GridMode.CUBE
        io_config["only_leaves"] = True
        io_config["overwrite_file"] = True
        grid.config = grid.config.model_copy(
            update={"io": IoConfig.model_validate(io_config)}
        )

        # Save the grid
        save_grid(grid, "test_cube_mode.vtkhdf")

        # Verify file and structure
        output_file = temp_output_dir / "test_cube_mode.vtkhdf"
        assert output_file.exists()

    def test_save_grid_file_exists_error(
        self, grid: TensorGrid, temp_output_dir: pathlib.Path, io_config: dict
    ) -> None:
        """Test save_grid raises FileExistsError when file exists."""
        # Update config with overwrite_file=False
        io_config["output_dir"] = temp_output_dir
        io_config["overwrite_file"] = False
        grid.config = grid.config.model_copy(
            update={"io": IoConfig.model_validate(io_config)}
        )

        # Create the file first
        output_file = temp_output_dir / "existing_file.vtkhdf"
        output_file.touch()

        # Test that FileExistsError is raised
        with pytest.raises(FileExistsError, match="File .* already exists"):
            save_grid(grid, "existing_file.vtkhdf")
