from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
PARAMETERS_PATH = ROOT / "data" / "parameters.yml"
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "slice_plots"

RE_DIR_PATTERN = re.compile(r"^re_(?P<re>.+)$")
VTU_STEP_PATTERN = re.compile(r"(\d+)$")


class CaseConfig(BaseModel):
    name: str
    mesh_path: str
    magU_ref: float
    A_ref: float
    L_ref: float


class ExperimentParameters(BaseModel):
    case: list[CaseConfig]
    Re: list[float]


@dataclass(frozen=True)
class SlicePlane:
    name: str
    normal: tuple[float, float, float]
    origin: tuple[float, float, float]
    view: str


SLICE_PLANES = {
    "xy": SlicePlane(
        name="xy",
        normal=(0.0, 0.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        view="view_xy",
    ),
    "xz": SlicePlane(
        name="xz",
        normal=(0.0, 1.0, 0.0),
        origin=(0.0, 0.0, 0.0),
        view="view_xz",
    ),
}


def _load_parameters(path: Path) -> ExperimentParameters:
    with path.open() as f:
        return ExperimentParameters.model_validate(yaml.safe_load(f))


def _re_label(re_value: float) -> str:
    return f"{re_value:g}"


def _gridfoam_output_dir(case_name: str, re_value: float) -> Path:
    return OUTPUTS_ROOT / case_name / "gridfoam" / f"re_{_re_label(re_value)}"


def _openfoam_output_dir(case_name: str, re_value: float) -> Path:
    return OUTPUTS_ROOT / case_name / "openfoam" / f"re_{_re_label(re_value)}"


def _parse_vtu_step(path: Path) -> int | None:
    match = VTU_STEP_PATTERN.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _find_final_gridfoam_vtu(case_name: str, re_value: float) -> Path:
    output_dir = _gridfoam_output_dir(case_name, re_value)
    candidates = [
        path
        for path in output_dir.glob(f"{case_name}*.vtu")
        if path.name != "surface_mesh.vtu"
    ]
    if not candidates:
        raise FileNotFoundError(
            f"gridfoam VTU output was not found under {output_dir}"
        )

    scored = [
        (step, path)
        for path in candidates
        if (step := _parse_vtu_step(path)) is not None
    ]
    if not scored:
        raise FileNotFoundError(
            f"gridfoam VTU files without step suffix were found under {output_dir}"
        )
    return max(scored, key=lambda item: item[0])[1]


def _gridfoam_vtu_path(
    case_name: str, re_value: float, time: int | None
) -> Path:
    if time is None:
        return _find_final_gridfoam_vtu(case_name, re_value)

    path = (
        _gridfoam_output_dir(case_name, re_value)
        / f"{case_name}{time:04d}.vtu"
    )
    if not path.exists():
        raise FileNotFoundError(f"gridfoam output was not found: {path}")
    return path


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


def _find_final_openfoam_time(case_dir: Path) -> int:
    time_dirs = [
        int(path.name)
        for path in case_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not time_dirs:
        raise FileNotFoundError(
            f"OpenFOAM time directories were not found under {case_dir}"
        )
    return max(time_dirs)


def _read_openfoam_mesh(case_dir: Path, time: int | None) -> pv.DataSet:
    if not case_dir.exists():
        raise FileNotFoundError(f"OpenFOAM case was not found: {case_dir}")

    if time is None:
        time = _find_final_openfoam_time(case_dir)

    foam_path = case_dir / "case.foam"
    foam_path.touch(exist_ok=True)
    reader = pv.OpenFOAMReader(str(foam_path))
    if float(time) not in reader.time_values:
        raise ValueError(
            f"OpenFOAM case has times {reader.time_values}, "
            f"but requested time={time}."
        )
    reader.set_active_time_value(float(time))
    return _largest_dataset(reader.read())


def _read_gridfoam_mesh(
    case_name: str, re_value: float, time: int | None
) -> pv.DataSet:
    path = _gridfoam_vtu_path(case_name, re_value, time)
    return pv.read(path)


def _prepare_slice(mesh: pv.DataSet, plane: SlicePlane) -> pv.DataSet:
    if mesh.n_cells == 0:
        raise ValueError("Cannot slice an empty mesh.")

    prepared = mesh.cell_data_to_point_data(pass_cell_data=True)
    sliced = prepared.slice(normal=plane.normal, origin=plane.origin)
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
    plane: SlicePlane,
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
    getattr(plotter, plane.view)()
    plotter.renderer.enable_parallel_projection()


def _plot_case(
    case_name: str,
    *,
    re_value: float,
    time: int | None,
    plane: SlicePlane,
    output_dir: Path,
    save_slices: bool,
    show_edges: bool,
    zoom: float,
) -> Path:
    gridfoam = _prepare_slice(
        _read_gridfoam_mesh(case_name, re_value, time),
        plane,
    )
    openfoam = _prepare_slice(
        _read_openfoam_mesh(_openfoam_output_dir(case_name, re_value), time),
        plane,
    )

    re_label = _re_label(re_value)
    if save_slices:
        gridfoam.save(
            output_dir
            / f"{case_name}_re_{re_label}_{plane.name}_gridfoam.vtp"
        )
        openfoam.save(
            output_dir
            / f"{case_name}_re_{re_label}_{plane.name}_openfoam.vtp"
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
        (1, 0, gridfoam, "U_mag", "gridfoam |U|", u_range, "viridis"),
        (1, 1, openfoam, "U_mag", "OpenFOAM |U|", u_range, "viridis"),
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
            plane=plane,
            show_edges=show_edges,
        )
    plotter.link_views()
    plotter.renderer.add_axes()
    for row in range(2):
        for col in range(2):
            plotter.subplot(row, col)
            plotter.camera.zoom(zoom)

    output = (
        output_dir / f"{case_name}_re_{re_label}_{plane.name}_p_U.png"
    )
    plotter.screenshot(str(output))
    plotter.close()
    return output


def _parse_args(case_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot gridfoam vs OpenFOAM slice comparisons from "
            "experiments/re_vs_cd outputs."
        )
    )
    parser.add_argument(
        "--cases",
        choices=case_names,
        default=case_names,
        nargs="+",
        help="Cases to plot.",
    )
    parser.add_argument(
        "--re", default=100.0, type=float, help="Reynolds number."
    )
    parser.add_argument(
        "--plane",
        choices=tuple(SLICE_PLANES),
        default="xy",
        help="Slice plane: xy (z=0) or xz (y=0).",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help="Output time/step. Default: final available step.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Directory for generated PNG files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--save-slices",
        action="store_true",
        help="Also save sampled slices as VTP files.",
    )
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Show mesh edges in the slice plots.",
    )
    parser.add_argument("--zoom", default=1.0, type=float, help="Camera zoom.")
    return parser.parse_args()


def main() -> None:
    parameters = _load_parameters(PARAMETERS_PATH)
    case_names = [case.name for case in parameters.case]
    args = _parse_args(case_names)
    plane = SLICE_PLANES[args.plane]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case_name in args.cases:
        output = _plot_case(
            case_name,
            re_value=args.re,
            time=args.time,
            plane=plane,
            output_dir=args.output_dir,
            save_slices=args.save_slices,
            show_edges=args.show_edges,
            zoom=args.zoom,
        )
        print(output)


if __name__ == "__main__":
    main()
