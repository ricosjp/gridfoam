from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent
CASES = ("sphere", "cylinder", "cube")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot y=0 p and |U| slice comparisons for low-Re gridfoam and "
            "OpenFOAM cases."
        )
    )
    parser.add_argument(
        "--cases",
        choices=CASES,
        default=CASES,
        nargs="+",
        help="Cases to plot.",
    )
    parser.add_argument(
        "--re", default=100.0, type=float, help="Reynolds number."
    )
    parser.add_argument(
        "--time", default=80, type=int, help="Output time/step."
    )
    parser.add_argument(
        "--output-dir",
        default=ROOT / "low_re_slice_plots",
        type=Path,
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--run-openfoam",
        action="store_true",
        help="Run each OpenFOAM case at --re before plotting.",
    )
    parser.add_argument(
        "--allow-openfoam-re-mismatch",
        action="store_true",
        help="Plot even if OpenFOAM transportProperties does not match --re.",
    )
    parser.add_argument(
        "--save-slices",
        action="store_true",
        help="Also save sampled y=0 slices as VTP files.",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Show mesh edges in the slice plots.",
    )
    parser.add_argument("--zoom", default=1.0, type=float, help="Camera zoom.")
    return parser.parse_args()


def _case_root(case: str) -> Path:
    return ROOT / f"low_re_{case}"


def _re_label(re_value: float) -> str:
    return f"{re_value:g}"


def _gridfoam_vtu_path(case: str, re_value: float, time: int) -> Path:
    name = f"low_re_{case}_{time:04d}.vtu"
    return (
        _case_root(case)
        / "gridfoam"
        / "outputs"
        / f"re_{_re_label(re_value)}"
        / name
    )


def _openfoam_root(case: str) -> Path:
    return _case_root(case) / "of"


def _read_openfoam_nu(case: str) -> float | None:
    path = _openfoam_root(case) / "constant" / "transportProperties"
    if not path.exists():
        return None
    text = path.read_text()
    pattern = r"^\s*nu\s+(?:\[[^\]]+\]\s*)?([0-9.eE+-]+)\s*;"
    match = re.search(pattern, text, re.M)
    return None if match is None else float(match.group(1))


def _check_openfoam_re(
    case: str, re_value: float, allow_mismatch: bool
) -> None:
    nu = _read_openfoam_nu(case)
    if nu is None:
        raise FileNotFoundError(
            f"Could not read OpenFOAM nu for low_re_{case}."
        )
    expected = 1.0 / re_value
    if np.isclose(nu, expected, rtol=1e-8, atol=1e-12):
        return
    message = (
        f"OpenFOAM low_re_{case} has nu={nu:g}, which corresponds to "
        f"Re={1.0 / nu:g}, not requested Re={re_value:g}. Run with "
        "--run-openfoam, or pass --allow-openfoam-re-mismatch to plot the "
        "current OpenFOAM fields anyway."
    )
    if allow_mismatch:
        print(f"warning: {message}")
        return
    raise RuntimeError(message)


def _run_openfoam_case(case: str, re_value: float) -> None:
    root = _openfoam_root(case)
    subprocess.run(
        [sys.executable, "run_sweep.py", "--re", _re_label(re_value)],
        cwd=root,
        check=True,
    )


def _iter_datasets(mesh: pv.DataObject):
    if isinstance(mesh, pv.MultiBlock):
        for block in mesh:
            if block is not None:
                yield from _iter_datasets(block)
    elif isinstance(mesh, pv.DataSet):
        yield mesh


def _largest_dataset(mesh: pv.DataObject) -> pv.DataSet:
    datasets = list(_iter_datasets(mesh))
    if not datasets:
        raise ValueError("No datasets were found in OpenFOAM reader output.")
    return max(datasets, key=lambda item: item.n_cells)


def _read_gridfoam_mesh(case: str, re_value: float, time: int) -> pv.DataSet:
    path = _gridfoam_vtu_path(case, re_value, time)
    if not path.exists():
        raise FileNotFoundError(f"gridfoam output was not found: {path}")
    return pv.read(path)


def _read_openfoam_mesh(case: str, time: int) -> pv.DataSet:
    foam_path = _openfoam_root(case) / "case.foam"
    foam_path.touch(exist_ok=True)
    reader = pv.OpenFOAMReader(str(foam_path))
    if float(time) not in reader.time_values:
        raise ValueError(
            f"OpenFOAM low_re_{case} has times {reader.time_values}, "
            f"but requested time={time}."
        )
    reader.set_active_time_value(float(time))
    return _largest_dataset(reader.read())


def _prepare_slice(mesh: pv.DataSet) -> pv.DataSet:
    if mesh.n_cells == 0:
        raise ValueError("Cannot slice an empty mesh.")
    prepared = mesh.cell_data_to_point_data(pass_cell_data=True)
    sliced = prepared.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0))
    if "U" not in sliced.point_data and "U" not in sliced.cell_data:
        raise KeyError("Field 'U' was not found in slice output.")
    U = (
        sliced.point_data["U"]
        if "U" in sliced.point_data
        else sliced.cell_data["U"]
    )
    sliced.point_data["U_mag"] = np.linalg.norm(U, axis=1)
    return sliced


def _field_values(mesh: pv.DataSet, field: str) -> np.ndarray:
    if field in mesh.point_data:
        values = mesh.point_data[field]
    elif field in mesh.cell_data:
        values = mesh.cell_data[field]
    else:
        raise KeyError(f"Field {field!r} was not found in slice output.")
    return np.asarray(values).reshape(-1)


def _field_range(meshes: list[pv.DataSet], field: str) -> tuple[float, float]:
    values = np.concatenate([_field_values(mesh, field) for mesh in meshes])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Field {field!r} has no finite values.")
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def _add_slice(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    *,
    field: str,
    title: str,
    clim: tuple[float, float],
    cmap: str,
    show_edges: bool,
) -> None:
    plotter.add_mesh(
        mesh,
        scalars=field,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        line_width=0.3,
        scalar_bar_args={"title": field, "vertical": True},
    )
    plotter.add_text(title, font_size=11)
    # plotter.view_xz()
    plotter.view_xy()
    plotter.enable_parallel_projection()


def _plot_case(
    case: str,
    *,
    re_value: float,
    time: int,
    output_dir: Path,
    save_slices: bool,
    show_edges: bool,
    zoom: float,
) -> Path:
    gridfoam = _prepare_slice(_read_gridfoam_mesh(case, re_value, time))
    openfoam = _prepare_slice(_read_openfoam_mesh(case, time))

    if save_slices:
        gridfoam.save(
            output_dir
            / f"low_re_{case}_re_{_re_label(re_value)}_gridfoam_z0.vtp"
        )
        openfoam.save(
            output_dir
            / f"low_re_{case}_re_{_re_label(re_value)}_openfoam_z0.vtp"
        )

    p_range = _field_range([gridfoam, openfoam], "p")
    u_range = _field_range([gridfoam, openfoam], "U_mag")

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True, shape=(2, 2), window_size=(1800, 1300)
    )
    panels = [
        (0, 0, gridfoam, "p", "gridfoam p", p_range, "coolwarm"),
        (0, 1, openfoam, "p", "OpenFOAM p", p_range, "coolwarm"),
        (1, 0, gridfoam, "U_mag", "gridfoam mag_U", u_range, "viridis"),
        (1, 1, openfoam, "U_mag", "OpenFOAM mag_U", u_range, "viridis"),
    ]
    for row, col, mesh, field, title, clim, cmap in panels:
        plotter.subplot(row, col)
        _add_slice(
            plotter,
            mesh,
            field=field,
            title=title,
            clim=clim,
            cmap=cmap,
            show_edges=show_edges,
        )
    plotter.link_views()
    plotter.show_axes()
    for row in range(2):
        for col in range(2):
            plotter.subplot(row, col)
            plotter.camera.zoom(zoom)

    output = output_dir / f"low_re_{case}_re_{_re_label(re_value)}_z0_p_U.png"
    plotter.screenshot(str(output))
    plotter.close()
    return output


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        if args.run_openfoam:
            _run_openfoam_case(case, args.re)
        _check_openfoam_re(case, args.re, args.allow_openfoam_re_mismatch)
        output = _plot_case(
            case,
            re_value=args.re,
            time=args.time,
            output_dir=args.output_dir,
            save_slices=args.save_slices,
            show_edges=args.show_edges,
            zoom=args.zoom,
        )
        print(output)


if __name__ == "__main__":
    main()
