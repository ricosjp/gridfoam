import pathlib

import pytest
import pyvista as pv
from pytest_benchmark.fixture import BenchmarkFixture

from gridfoam._geometry import TriangleMesh
from gridfoam._octree import Forest
from gridfoam.settings import GridSetting


@pytest.fixture
def grid_setting_bunny() -> GridSetting:
    """Return a Settings object."""
    return GridSetting(
        blockXMin=-4.0,
        blockXMax=4.0,
        blockYMin=-2.0,
        blockYMax=2.0,
        blockZMin=-2.0,
        blockZMax=2.0,
        nBlockX=8,
        nBlockY=4,
        nBlockZ=4,
        alpha=0.3,
        level_limit=7,
    )


@pytest.mark.with_benchmark
@pytest.mark.parametrize("level_limit", [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_gridgen_bunny_benchmark_time(
    benchmark: BenchmarkFixture,
    grid_setting_bunny: GridSetting,
    level_limit: int,
):
    """Test the grid generation process."""
    benchmark_grid_setting_bunny = grid_setting_bunny.model_copy(
        update={"level_limit": level_limit}
    )

    def gridgen_process() -> None:
        pv_mesh = pv.read("tests/data/stl/bunny.stl")
        mesh = TriangleMesh.from_polydata(pv_mesh)
        forest = Forest(benchmark_grid_setting_bunny)
        grid = forest.build_grid_from_mesh(mesh)
        file_name = pathlib.Path(
            f"tests/outputs/grid/bunny_level{level_limit}.vtkhdf"
        )
        grid.save_structure(file_name)

    benchmark(gridgen_process)


@pytest.fixture
def grid_setting_DrivAer() -> GridSetting:
    """Return a Settings object."""
    return GridSetting(
        blockXMin=-5.0,
        blockXMax=11.0,
        blockYMin=-4.0,
        blockYMax=4.0,
        blockZMin=0.0,
        blockZMax=4.0,
        nBlockX=8,
        nBlockY=4,
        nBlockZ=2,
        alpha=0.3,
        level_limit=3,
    )


@pytest.mark.with_benchmark
def test_gridgen_DrivAer_benchmark_time(
    benchmark: BenchmarkFixture, grid_setting_DrivAer: GridSetting
):
    """Test the grid generation process."""

    def gridgen_process() -> None:
        pv_mesh = pv.read("tests/data/stl/DrivAer.stl")
        mesh = TriangleMesh.from_polydata(pv_mesh)
        forest = Forest(grid_setting_DrivAer)
        grid = forest.build_grid_from_mesh(mesh)
        file_name = pathlib.Path("tests/outputs/grid/DrivAer.vtkhdf")
        grid.save_structure(file_name)

    benchmark(gridgen_process)
