import pathlib

import pytest
import pyvista as pv

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
def test_grid_generation_process_bunny(grid_setting_bunny: GridSetting):
    """Test the grid generation process."""
    pv_mesh = pv.read("tests/data/stl/bunny.stl")
    mesh = TriangleMesh.from_polydata(pv_mesh)
    forest = Forest(grid_setting_bunny)
    grid = forest.build_grid_from_mesh(mesh)
    file_name = pathlib.Path("tests/outputs/grid/bunny.vtkhdf")
    grid.save_structure(file_name)

    assert file_name.exists()


# @pytest.fixture
# def grid_setting_DrivAer() -> GridSetting:
#     """Return a Settings object."""
#     return GridSetting(
#         blockXMin=-5.0,
#         blockXMax=11.0,
#         blockYMin=-4.0,
#         blockYMax=4.0,
#         blockZMin=0.0,
#         blockZMax=4.0,
#         nBlockX=8,
#         nBlockY=4,
#         nBlockZ=2,
#         alpha=0.3,
#         level_limit=3,
#     )

# def test_grid_generation_process_DrivAer(grid_setting_DrivAer: GridSetting):
#     """Test the grid generation process."""
#     pv_mesh = pv.read("tests/data/stl/DrivAer.stl")
#     mesh = TriangleMesh.from_polydata(pv_mesh)
#     forest = Forest(grid_setting_DrivAer)
#     grid = forest.build_grid_from_mesh(mesh)
#     file_name = pathlib.Path("tests/outputs/grid/DrivAer.vtkhdf")
#     grid.save_structure(file_name)

#     assert file_name.exists()
