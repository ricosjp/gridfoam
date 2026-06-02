from __future__ import annotations

from pathlib import Path

import pyvista as pv


def write_cube(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cube = pv.Box(
        bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
        level=0,
        quads=False,
    ).triangulate()
    cube.compute_normals(
        point_normals=False,
        cell_normals=True,
        auto_orient_normals=True,
        inplace=True,
    )
    cube.save(path)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    write_cube(root / "gridfoam" / "data" / "cube.stl")
    write_cube(root / "of" / "constant" / "triSurface" / "cube.stl")


if __name__ == "__main__":
    main()

