from __future__ import annotations

from pathlib import Path

import pyvista as pv


def write_cylinder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cylinder = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=0.5,
        height=8.0,
        resolution=96,
        capping=True,
    ).triangulate()
    cylinder.compute_normals(
        point_normals=False,
        cell_normals=True,
        auto_orient_normals=True,
        inplace=True,
    )
    cylinder.save(path)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    write_cylinder(root / "gridfoam" / "data" / "cylinder.stl")
    write_cylinder(root / "of" / "constant" / "triSurface" / "cylinder.stl")


if __name__ == "__main__":
    main()

