import pathlib
import tempfile

import pytest
import yaml

from gridfoam import TensorGrid, save_grid


@pytest.mark.with_profile
def test_gridgen_bunny_profile():
    """Test the grid generation process."""
    # Load bunny config
    config_path = pathlib.Path("tests/data/yaml/bunny.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Modify depth_limit
    config_dict["octree"]["refinement"]["depth_limit"] = 8

    # Create temporary config file with modified depth_limit
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config_dict, f)
        temp_config_path = pathlib.Path(f.name)

    def gridgen_process() -> None:
        grid = TensorGrid.build(temp_config_path)
        save_grid(grid)

    try:
        gridgen_process()
    finally:
        temp_config_path.unlink()



@pytest.mark.with_profile
def test_gridgen_DrivAer_profile():
    """Test the grid generation process."""
    # Load DrivAer config
    config_path = pathlib.Path("tests/data/yaml/DrivAer.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Modify depth_limit
    config_dict["octree"]["refinement"]["depth_limit"] = 8

    # Create temporary config file with modified depth_limit
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config_dict, f)
        temp_config_path = pathlib.Path(f.name)

    def gridgen_process() -> None:
        grid = TensorGrid.build(temp_config_path)
        save_grid(grid)

    try:
        gridgen_process()
    finally:
        temp_config_path.unlink()
