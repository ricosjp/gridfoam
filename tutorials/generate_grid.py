import pathlib

import pyvista as pv

from gridfoam._geometry import TriangleMesh
from gridfoam._octree import Forest
from gridfoam.settings import GridSetting


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


def generate_grid_DrivAer(
    grid_setting: GridSetting,
    depth_limit: int,
):
    """Test the grid generation process."""

    pv_mesh = pv.read("tests/data/stl/DrivAer.stl")
    mesh = TriangleMesh.from_polydata(pv_mesh)
    forest = Forest(grid_setting)
    grid = forest.build_grid_from_mesh(mesh)
    file_name = pathlib.Path(f"DrivAer_depth{depth_limit}.vtkhdf")
    # grid.save_structure(file_name, only_leaves=False)
    grid.save_grid(file_name, only_leaves=True)


if __name__ == "__main__":
    generate_grid_DrivAer(grid_setting_DrivAer(), 2)
