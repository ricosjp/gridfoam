import pathlib
import tempfile
from collections.abc import Callable

import pytest
import yaml
from pytest_benchmark.fixture import BenchmarkFixture

from gridfoam import TensorGrid, save_grid


def benchmark_with_group(func: Callable) -> Callable:
    return pytest.mark.benchmark(group=func.__name__)(func)


@pytest.mark.with_benchmark
@pytest.mark.parametrize("depth_limit", [2, 3, 4, 5, 6, 7, 8])
@benchmark_with_group
def test_gridgen_bunny(
    benchmark: BenchmarkFixture,
    depth_limit: int,
):
    """Test the grid generation process."""
    # Load bunny config
    config_path = pathlib.Path("tests/data/yaml/bunny.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Modify depth_limit
    config_dict["octree"]["refinement"]["depth_limit"] = depth_limit
    config_dict["io"]["mode"] = "cube"

    # Create temporary config file with modified depth_limit
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config_dict, f)
        temp_config_path = pathlib.Path(f.name)

    def gridgen_process() -> None:
        grid = TensorGrid.build(temp_config_path)
        save_grid(grid, "test.vtkhdf")

    try:
        benchmark(gridgen_process)
    finally:
        temp_config_path.unlink()


@pytest.mark.with_benchmark
@pytest.mark.parametrize("depth_limit", [2, 3, 4, 5, 6, 7, 8])
@benchmark_with_group
def test_gridgen_DrivAer(
    benchmark: BenchmarkFixture,
    depth_limit: int,
):
    """Test the grid generation process."""
    # Load DrivAer config
    config_path = pathlib.Path("tests/data/yaml/DrivAer.yaml")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Modify depth_limit
    config_dict["octree"]["refinement"]["depth_limit"] = depth_limit
    config_dict["io"]["mode"] = "cube"

    # Create temporary config file with modified depth_limit
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config_dict, f)
        temp_config_path = pathlib.Path(f.name)

    def gridgen_process() -> None:
        grid = TensorGrid.build(temp_config_path)
        save_grid(grid, "test.vtkhdf")

    try:
        benchmark(gridgen_process)
    finally:
        temp_config_path.unlink()
