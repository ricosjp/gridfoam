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


def write_sphere(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sphere = pv.Sphere(
        radius=0.5,
        center=(0.0, 0.0, 0.0),
        theta_resolution=64,
        phi_resolution=32,
    ).triangulate()
    sphere.compute_normals(
        point_normals=False,
        cell_normals=True,
        auto_orient_normals=True,
        inplace=True,
    )
    sphere.save(path)

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    write_cylinder(root / "re_vs_cd" / "data" / "cylinder.stl")
    write_sphere(root / "re_vs_cd" / "data" / "sphere.stl")


if __name__ == "__main__":
    main()
