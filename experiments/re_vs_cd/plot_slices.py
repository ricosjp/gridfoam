from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyvista as pv
import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
PARAMETERS_PATH = ROOT / "data" / "parameters.yml"
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "slice_plots"

VTU_STEP_PATTERN = re.compile(r"(\d+)$")

SourceName = Literal["gridfoam", "openfoam", "openfoam_fine"]

SOURCE_LABELS: dict[SourceName, str] = {
    "gridfoam": "gridfoam",
    "openfoam": "OpenFOAM",
    "openfoam_fine": "OpenFOAM fine",
}
SOURCE_DIRS: dict[SourceName, str] = {
    "gridfoam": "gridfoam",
    "openfoam": "openfoam",
    "openfoam_fine": "openfoam_fine",
}
COMPARE_PRESETS: dict[str, tuple[SourceName, SourceName]] = {
    "gridfoam-openfoam": ("gridfoam", "openfoam"),
    "gridfoam-openfoam_fine": ("gridfoam", "openfoam_fine"),
    "openfoam-openfoam_fine": ("openfoam", "openfoam_fine"),
}


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


def _source_output_dir(
    source: SourceName, case_name: str, re_value: float
) -> Path:
    return (
        OUTPUTS_ROOT
        / case_name
        / SOURCE_DIRS[source]
        / (f"re_{_re_label(re_value)}")
    )


def _parse_vtu_step(path: Path) -> int | None:
    match = VTU_STEP_PATTERN.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _find_final_gridfoam_vtu(output_dir: Path, case_name: str) -> Path:
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
            "gridfoam VTU files without step suffix were found under "
            f"{output_dir}"
        )
    return max(scored, key=lambda item: item[0])[1]


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
    output_dir = _source_output_dir("gridfoam", case_name, re_value)
    if time is None:
        path = _find_final_gridfoam_vtu(output_dir, case_name)
    else:
        path = output_dir / f"{case_name}_{time:04d}.vtu"
        if not path.exists():
            raise FileNotFoundError(f"gridfoam output was not found: {path}")
    return pv.read(path)


def _read_source_mesh(
    source: SourceName,
    case_name: str,
    re_value: float,
    time: int | None,
) -> pv.DataSet:
    if source == "gridfoam":
        return _read_gridfoam_mesh(case_name, re_value, time)
    return _read_openfoam_mesh(
        _source_output_dir(source, case_name, re_value), time
    )


def _list_source_times(
    source: SourceName, case_name: str, re_value: float
) -> list[int]:
    output_dir = _source_output_dir(source, case_name, re_value)
    if source == "gridfoam":
        times = [
            step
            for path in output_dir.glob(f"{case_name}*.vtu")
            if path.name != "surface_mesh.vtu"
            and (step := _parse_vtu_step(path)) is not None
        ]
        return sorted(set(times))

    if not output_dir.exists():
        raise FileNotFoundError(f"OpenFOAM case was not found: {output_dir}")
    times = [
        int(path.name)
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and int(path.name) > 0
    ]
    return sorted(set(times))


def _average_fields(
    meshes: list[pv.DataSet], fields: tuple[str, ...]
) -> pv.DataSet:
    """Return a copy of the first mesh with selected cell fields averaged."""
    if not meshes:
        raise ValueError("Cannot average an empty mesh list.")

    averaged = meshes[0].copy(deep=True)
    n = len(meshes)
    for field in fields:
        if field not in averaged.cell_data:
            raise KeyError(
                f"Field {field!r} was not found while averaging meshes."
            )
        total = np.asarray(averaged.cell_data[field], dtype=np.float64)
        for mesh in meshes[1:]:
            if field not in mesh.cell_data:
                raise KeyError(
                    f"Field {field!r} was not found while averaging meshes."
                )
            values = np.asarray(mesh.cell_data[field], dtype=np.float64)
            if values.shape != total.shape:
                raise ValueError(
                    f"Field {field!r} shape mismatch while averaging: "
                    f"{total.shape} vs {values.shape}"
                )
            total += values
        averaged.cell_data[field] = (total / n).astype(
            np.asarray(averaged.cell_data[field]).dtype, copy=False
        )
    return averaged


def _read_averaged_source_mesh(
    source: SourceName,
    case_name: str,
    re_value: float,
    times: list[int],
) -> pv.DataSet:
    if not times:
        raise ValueError(f"No times available to average for source={source!r}")
    meshes = [
        _read_source_mesh(source, case_name, re_value, time) for time in times
    ]
    return _average_fields(meshes, fields=("p", "U"))


def _select_average_times(
    source: SourceName,
    case_name: str,
    re_value: float,
    average_from: int,
    average_to: int,
) -> list[int]:
    available = _list_source_times(source, case_name, re_value)
    selected = [t for t in available if average_from <= t <= average_to]
    if not selected:
        raise ValueError(
            f"No write times in [{average_from}, {average_to}] for "
            f"{source}/{case_name}/re_{_re_label(re_value)}; "
            f"available={available}"
        )
    return selected


def _load_comparison_meshes(
    case_name: str,
    *,
    re_value: float,
    time: int | None,
    average_from: int | None,
    average_to: int | None,
    left_source: SourceName,
    right_source: SourceName,
) -> tuple[pv.DataSet, pv.DataSet, str]:
    if average_from is None and average_to is None:
        return (
            _read_source_mesh(left_source, case_name, re_value, time),
            _read_source_mesh(right_source, case_name, re_value, time),
            "",
        )
    if average_from is None or average_to is None:
        raise ValueError(
            "Both --average-from and --average-to must be set together."
        )
    if time is not None:
        raise ValueError("Use either --time or --average-from/--average-to.")
    if average_from > average_to:
        raise ValueError(
            f"--average-from ({average_from}) must be <= "
            f"--average-to ({average_to})."
        )

    left_times = _select_average_times(
        left_source, case_name, re_value, average_from, average_to
    )
    right_times = _select_average_times(
        right_source, case_name, re_value, average_from, average_to
    )
    print(
        f"{case_name} re={_re_label(re_value)}: "
        f"averaging {left_source} times={left_times}, "
        f"{right_source} times={right_times}"
    )
    return (
        _read_averaged_source_mesh(
            left_source, case_name, re_value, left_times
        ),
        _read_averaged_source_mesh(
            right_source, case_name, re_value, right_times
        ),
        f"_avg_{average_from}_{average_to}",
    )


def _error_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite values available for error statistics.")
    return {
        "mean_abs": float(np.mean(np.abs(finite))),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "max_abs": float(np.max(np.abs(finite))),
    }


def _rename_fields(mesh: pv.DataSet, prefix: str) -> None:
    """Prefix all point/cell data arrays to avoid name collisions."""
    for data in (mesh.point_data, mesh.cell_data):
        for name in list(data.keys()):
            data[f"{prefix}{name}"] = data.pop(name)


def _lookup_field(mesh: pv.DataSet, field: str) -> np.ndarray:
    if field in mesh.point_data:
        return np.asarray(mesh.point_data[field])
    if field in mesh.cell_data:
        return np.asarray(mesh.cell_data[field])
    raise KeyError(f"Field {field!r} was not found in slice output.")


def _field_values(mesh: pv.DataSet, field: str) -> np.ndarray:
    return _lookup_field(mesh, field).reshape(-1)


def _vector_field_values(mesh: pv.DataSet, field: str) -> np.ndarray:
    values = _lookup_field(mesh, field)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values.reshape(values.shape[0], -1)


def _masked_values(
    sampled: pv.DataSet,
    field: str,
    valid_mask: np.ndarray,
    *,
    vector: bool = False,
) -> np.ndarray:
    getter = _vector_field_values if vector else _field_values
    values = getter(sampled, field).astype(np.float64, copy=True)
    values[~valid_mask] = np.nan
    return values


def _overlap_bounds_2d(
    left: pv.DataSet, right: pv.DataSet, plane: SlicePlane
) -> tuple[float, float, float, float]:
    """Return in-plane overlap of two 3D volume bounds."""
    lb = left.bounds
    rb = right.bounds
    if plane.name == "xy":
        xmin = max(lb[0], rb[0])
        xmax = min(lb[1], rb[1])
        ymin = max(lb[2], rb[2])
        ymax = min(lb[3], rb[3])
    elif plane.name == "xz":
        xmin = max(lb[0], rb[0])
        xmax = min(lb[1], rb[1])
        ymin = max(lb[4], rb[4])
        ymax = min(lb[5], rb[5])
    else:
        raise ValueError(f"Unsupported plane: {plane.name!r}")

    if not (xmin < xmax and ymin < ymax):
        raise ValueError(
            "No geometric overlap between volumes for uniform sampling."
        )
    return xmin, xmax, ymin, ymax


def _build_uniform_plane_grid(
    plane: SlicePlane,
    bounds_2d: tuple[float, float, float, float],
    *,
    nx: int,
    ny: int,
) -> pv.StructuredGrid:
    xmin, xmax, ymin, ymax = bounds_2d
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    if plane.name == "xy":
        zz = np.full_like(xx, plane.origin[2], dtype=np.float64)
        return pv.StructuredGrid(xx, yy, zz)
    if plane.name == "xz":
        yy_const = np.full_like(xx, plane.origin[1], dtype=np.float64)
        return pv.StructuredGrid(xx, yy_const, yy)
    raise ValueError(f"Unsupported plane: {plane.name!r}")


def _valid_point_mask(sampled: pv.DataSet, n_points: int) -> np.ndarray:
    mask = sampled.point_data.get(
        "vtkValidPointMask", np.ones(n_points, dtype=np.uint8)
    )
    return np.asarray(mask).astype(bool)


def _sample_volume_on_plane(
    volume: pv.DataSet, probe: pv.StructuredGrid
) -> pv.DataSet:
    """Sample a 3D volume at probe points by querying enclosing 3D cells.

    Probe points lie on the comparison plane, but interpolation uses the
    source cell that contains each 3D point rather than a 2D slice.
    """
    sampled = probe.sample(
        volume,
        pass_cell_data=False,
        pass_point_data=True,
        locator="cell_tree",
    )
    if not isinstance(sampled, pv.DataSet):
        raise TypeError("3D sampling did not return a DataSet.")
    return sampled


def _has_disconnected_cell_points(mesh: pv.DataSet) -> bool:
    """Return True if each cell stores its own unshared vertices."""
    if mesh.n_cells == 0:
        return False
    return mesh.n_points >= 7 * mesh.n_cells


def _ensure_point_data(mesh: pv.DataSet, fields: tuple[str, ...]) -> pv.DataSet:
    """Build shared-vertex point data so 3D probing interpolates smoothly.

    gridfoam VTUs typically emit one disjoint hex per cell (8 points/cell).
    Averaging cell data onto those duplicated vertices leaves a piecewise
    constant field. Merge coincident points first, then interpolate.
    """
    prepared = mesh
    disconnected = _has_disconnected_cell_points(prepared)
    if disconnected:
        # merge_points() keeps orphan vertex cells and drops arrays on
        # gridfoam VTUs; clean() merges coincident points in-place.
        prepared = prepared.clean(tolerance=1e-9)

    missing_point_data = any(
        field in prepared.cell_data and field not in prepared.point_data
        for field in fields
    )
    if disconnected or missing_point_data:
        converted = prepared.cell_data_to_point_data(pass_cell_data=False)
        if not isinstance(converted, pv.DataSet):
            raise TypeError("cell_data_to_point_data did not return a DataSet.")
        prepared = converted
    for field in fields:
        if field in prepared.cell_data:
            del prepared.cell_data[field]
    return prepared


def _extract_valid_region(mesh: pv.DataSet) -> pv.DataSet:
    """Drop invalid probe points so body/boundary holes render as white."""
    if "valid_mask" not in mesh.point_data:
        return mesh
    mask = np.asarray(mesh.point_data["valid_mask"]).reshape(-1).astype(bool)
    if mask.size != mesh.n_points or bool(mask.all()):
        return mesh
    extracted = mesh.extract_points(
        mask, adjacent_cells=False, include_cells=True
    )
    if extracted.n_cells == 0:
        return mesh
    return extracted


def _build_comparison_slice(
    left_mesh: pv.DataSet,
    right_mesh: pv.DataSet,
    plane: SlicePlane,
    *,
    left_prefix: str = "left_",
    right_prefix: str = "right_",
    uniform_nx: int = 480,
    uniform_ny: int = 180,
) -> pv.DataSet:
    left = left_mesh.copy(deep=True)
    right = right_mesh.copy(deep=True)
    _rename_fields(left, left_prefix)
    _rename_fields(right, right_prefix)

    left_p = f"{left_prefix}p"
    right_p = f"{right_prefix}p"
    left_u = f"{left_prefix}U"
    right_u = f"{right_prefix}U"
    left = _ensure_point_data(left, (left_p, left_u))
    right = _ensure_point_data(right, (right_p, right_u))

    bounds_2d = _overlap_bounds_2d(left, right, plane)
    uniform = _build_uniform_plane_grid(
        plane, bounds_2d, nx=uniform_nx, ny=uniform_ny
    )
    left_sampled = _sample_volume_on_plane(left, uniform)
    right_sampled = _sample_volume_on_plane(right, uniform)

    required = (
        (left_p, left_sampled),
        (right_p, right_sampled),
        (left_u, left_sampled),
        (right_u, right_sampled),
    )
    missing = [
        field
        for field, mesh in required
        if field not in mesh.point_data and field not in mesh.cell_data
    ]
    if missing:
        raise KeyError(
            "Field(s) "
            + ", ".join(repr(name) for name in missing)
            + " were not found after 3D sampling."
        )

    left_valid = _valid_point_mask(left_sampled, uniform.n_points)
    right_valid = _valid_point_mask(right_sampled, uniform.n_points)
    valid = left_valid & right_valid
    lp = _masked_values(left_sampled, left_p, valid)
    rp = _masked_values(right_sampled, right_p, valid)
    lu = _masked_values(left_sampled, left_u, valid, vector=True)
    ru = _masked_values(right_sampled, right_u, valid, vector=True)
    lu_mag = np.linalg.norm(lu, axis=1)
    ru_mag = np.linalg.norm(ru, axis=1)

    comparison = uniform.copy(deep=True)
    comparison.point_data[left_p] = lp
    comparison.point_data[right_p] = rp
    comparison.point_data["p_diff"] = np.abs(lp - rp)
    comparison.point_data[f"{left_prefix}U_mag"] = lu_mag
    comparison.point_data[f"{right_prefix}U_mag"] = ru_mag
    comparison.point_data["U_mag_diff"] = np.abs(lu_mag - ru_mag)
    comparison.point_data["valid_mask"] = valid.astype(np.uint8)
    return _extract_valid_region(comparison)


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


def _union_range(mesh: pv.DataSet, *fields: str) -> tuple[float, float]:
    ranges = [_field_range([mesh], field) for field in fields]
    return min(item[0] for item in ranges), max(item[1] for item in ranges)


def _comparison_clims(
    mesh: pv.DataSet,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Return shared color limits for p, |U|, |Δp|, and |Δ|U||."""
    return (
        _union_range(mesh, "left_p", "right_p"),
        _union_range(mesh, "left_U_mag", "right_U_mag"),
        (0.0, _field_range([mesh], "p_diff")[1]),
        (0.0, _field_range([mesh], "U_mag_diff")[1]),
    )


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
        preference="point",
        interpolate_before_map=True,
        nan_color="white",
        scalar_bar_args={
            "title": field,
            "vertical": True,
            "fmt": "%.3f",
            "n_labels": 5,
        },
    )
    plotter.set_background("white")
    plotter.add_text(title, font_size=11)
    getattr(plotter, plane.view)()
    plotter.renderer.enable_parallel_projection()


def _comparison_output_path(
    output_dir: Path,
    case_name: str,
    re_label: str,
    plane_name: str,
    compare_tag: str,
    average_tag: str,
    left_source: SourceName,
    right_source: SourceName,
) -> Path:
    output = output_dir / (
        f"{case_name}_re_{re_label}_{plane_name}_{compare_tag}"
        f"{average_tag}_p_U.png"
    )
    # Keep the historical filename for the default instantaneous
    # gridfoam vs OpenFOAM plot.
    if (
        left_source == "gridfoam"
        and right_source == "openfoam"
        and not average_tag
    ):
        return output_dir / f"{case_name}_re_{re_label}_{plane_name}_p_U.png"
    return output


def _save_comparison_png(
    comparison: pv.DataSet,
    *,
    plane: SlicePlane,
    left_label: str,
    right_label: str,
    p_range: tuple[float, float],
    u_range: tuple[float, float],
    p_diff_range: tuple[float, float],
    u_diff_range: tuple[float, float],
    output: Path,
    show_edges: bool,
    zoom: float,
) -> None:
    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True, shape=(2, 3), window_size=(2400, 1300)
    )
    panels = (
        (0, 0, "left_p", f"{left_label} p", p_range, "coolwarm"),
        (0, 1, "right_p", f"{right_label} p", p_range, "coolwarm"),
        (0, 2, "p_diff", "|Δp|", p_diff_range, "magma"),
        (1, 0, "left_U_mag", f"{left_label} |U|", u_range, "viridis"),
        (1, 1, "right_U_mag", f"{right_label} |U|", u_range, "viridis"),
        (1, 2, "U_mag_diff", "|Δ|U||", u_diff_range, "magma"),
    )
    for row, col, field, title, clim, cmap in panels:
        plotter.subplot(row, col)
        _add_slice(
            plotter,
            comparison,
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
        for col in range(3):
            plotter.subplot(row, col)
            plotter.camera.zoom(zoom)
    plotter.screenshot(str(output))
    plotter.close()


def _plot_case(
    case_name: str,
    *,
    re_value: float,
    time: int | None,
    average_from: int | None,
    average_to: int | None,
    plane: SlicePlane,
    left_source: SourceName,
    right_source: SourceName,
    output_dir: Path,
    save_slices: bool,
    show_edges: bool,
    zoom: float,
    uniform_nx: int,
    uniform_ny: int,
) -> Path:
    left_mesh, right_mesh, average_tag = _load_comparison_meshes(
        case_name,
        re_value=re_value,
        time=time,
        average_from=average_from,
        average_to=average_to,
        left_source=left_source,
        right_source=right_source,
    )
    comparison = _build_comparison_slice(
        left_mesh,
        right_mesh,
        plane,
        uniform_nx=uniform_nx,
        uniform_ny=uniform_ny,
    )

    re_label = _re_label(re_value)
    compare_tag = f"{left_source}_vs_{right_source}"
    p_stats = _error_stats(_field_values(comparison, "p_diff"))
    u_stats = _error_stats(_field_values(comparison, "U_mag_diff"))
    print(
        f"{case_name} re={re_label} {compare_tag}{average_tag}: "
        f"|Δp| mean={p_stats['mean_abs']:.6e} "
        f"rms={p_stats['rms']:.6e} max={p_stats['max_abs']:.6e}; "
        f"|Δ|U|| mean={u_stats['mean_abs']:.6e} "
        f"rms={u_stats['rms']:.6e} max={u_stats['max_abs']:.6e}"
    )

    if save_slices:
        comparison.save(
            output_dir
            / (
                f"{case_name}_re_{re_label}_{plane.name}_{compare_tag}"
                f"{average_tag}.vtp"
            )
        )

    if average_tag:
        # Keep colorbars aligned with the instantaneous comparison plot.
        clims = _comparison_clims(
            _build_comparison_slice(
                _read_source_mesh(left_source, case_name, re_value, None),
                _read_source_mesh(right_source, case_name, re_value, None),
                plane,
                uniform_nx=uniform_nx,
                uniform_ny=uniform_ny,
            )
        )
    else:
        clims = _comparison_clims(comparison)

    output = _comparison_output_path(
        output_dir,
        case_name,
        re_label,
        plane.name,
        compare_tag,
        average_tag,
        left_source,
        right_source,
    )
    _save_comparison_png(
        comparison,
        plane=plane,
        left_label=SOURCE_LABELS[left_source],
        right_label=SOURCE_LABELS[right_source],
        p_range=clims[0],
        u_range=clims[1],
        p_diff_range=clims[2],
        u_diff_range=clims[3],
        output=output,
        show_edges=show_edges,
        zoom=zoom,
    )
    return output


def _parse_args(case_names: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot slice comparisons from experiments/re_vs_cd outputs."
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
        "--average-from",
        type=int,
        default=None,
        help=(
            "Start iteration/time for field averaging (inclusive). "
            "Must be used with --average-to."
        ),
    )
    parser.add_argument(
        "--average-to",
        type=int,
        default=None,
        help=(
            "End iteration/time for field averaging (inclusive). "
            "Must be used with --average-from."
        ),
    )
    parser.add_argument(
        "--compare",
        choices=tuple(COMPARE_PRESETS),
        default="gridfoam-openfoam",
        help=(
            "Comparison pair. Fields are sampled onto a shared uniform "
            "plane grid by querying enclosing 3D cells."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=(
            f"Directory for generated PNG files. Default: {DEFAULT_OUTPUT_DIR}"
        ),
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
    parser.add_argument(
        "--uniform-nx",
        default=480,
        type=int,
        help="Uniform sampling resolution in the first in-plane axis.",
    )
    parser.add_argument(
        "--uniform-ny",
        default=180,
        type=int,
        help="Uniform sampling resolution in the second in-plane axis.",
    )
    parser.add_argument("--zoom", default=1.0, type=float, help="Camera zoom.")
    return parser.parse_args()


def main() -> None:
    parameters = _load_parameters(PARAMETERS_PATH)
    case_names = [case.name for case in parameters.case]
    args = _parse_args(case_names)
    plane = SLICE_PLANES[args.plane]
    left_source, right_source = COMPARE_PRESETS[args.compare]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for case_name in args.cases:
        output = _plot_case(
            case_name,
            re_value=args.re,
            time=args.time,
            average_from=args.average_from,
            average_to=args.average_to,
            plane=plane,
            left_source=left_source,
            right_source=right_source,
            output_dir=args.output_dir,
            save_slices=args.save_slices,
            show_edges=args.show_edges,
            zoom=args.zoom,
            uniform_nx=args.uniform_nx,
            uniform_ny=args.uniform_ny,
        )
        print(output)


if __name__ == "__main__":
    main()
