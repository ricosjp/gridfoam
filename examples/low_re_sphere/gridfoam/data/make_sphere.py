from __future__ import annotations

from pathlib import Path

import pyvista as pv


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
    root = Path(__file__).resolve().parents[2]
    write_sphere(root / "gridfoam" / "data" / "sphere.stl")
    write_sphere(root / "of" / "constant" / "triSurface" / "sphere.stl")


if __name__ == "__main__":
    main()
