import json
import subprocess
from enum import StrEnum
from pathlib import Path

import matplotlib.colors as mcolors
import mlflow
import numpy as np
import pyvista as pv
import yaml
from pydantic import BaseModel, ConfigDict

PARAMETERS_PATH = Path("experiments/gf_vs_of/parameters.yml")
FAST_CMAP_PATH: Path = Path("experiments/utils/fast.json")


def _paraview_preset_to_cmap() -> mcolors.LinearSegmentedColormap:
    with open(FAST_CMAP_PATH) as f:
        presets = json.load(f)
    preset = presets[0] if isinstance(presets, list) else presets
    rgb = np.array(preset["RGBPoints"]).reshape(-1, 4)

    x = rgb[:, 0]
    colors = rgb[:, 1:4]
    name = preset["Name"]

    return mcolors.LinearSegmentedColormap.from_list(
        name, list(zip(x, colors, strict=True))
    )


FAST_CMAP = _paraview_preset_to_cmap()
EDGE_COLOR = "black"
EDGE_LINE_WIDTH = 0.1
EDGE_OPACITY = 0.08

class SliceMode(StrEnum):
    XY = "xy"
    XZ = "xz"
    YZ = "yz"


class CaseConfig(BaseModel):
    name: str
    slice_modes: list[SliceMode]
    slice_origin: list[float]
    openfoam: bool
    gridfoam: bool

    def get_slice_normal(self, mode: SliceMode) -> list[float]:
        match mode:
            case SliceMode.XY:
                return [0.0, 0.0, 1.0]
            case SliceMode.XZ:
                return [0.0, 1.0, 0.0]
            case SliceMode.YZ:
                return [1.0, 0.0, 0.0]

    def get_case_dir(self) -> Path:
        return Path("examples") / self.name

    def get_gridfoam_case_dir(self) -> Path:
        return self.get_case_dir() / "gridfoam"

    def get_openfoam_case_dir(self) -> Path:
        return self.get_case_dir() / "openfoam"


class ExperimentParameters(BaseModel):
    case: list[CaseConfig]


class PlotItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    cmap: str | mcolors.LinearSegmentedColormap


class PlotRow(BaseModel):
    ref_field: PlotItem
    our_field: PlotItem
    err_mag_field: PlotItem
    err_theta_field: PlotItem | None = None


def load_experiment_parameters(yaml_path: Path) -> ExperimentParameters:
    with yaml_path.open() as f:
        return ExperimentParameters.model_validate(yaml.safe_load(f))


def rename_fields(mesh: pv.DataSet, prefix: str) -> None:
    """Prefix all point/cell data arrays to avoid name collisions."""
    for data in (mesh.point_data, mesh.cell_data):
        for name in list(data.keys()):
            data[f"{prefix}{name}"] = data.pop(name)


def map_fields(source: pv.DataSet, target: pv.DataSet) -> None:
    """Interpolate source fields onto target cell centers."""
    probed = target.cell_centers().sample(source)
    for name in source.cell_data.keys():
        if name in probed.point_data:
            target.cell_data[name] = np.asarray(probed.point_data[name])


def slice_mesh(
    mesh: pv.DataSet,
    origin: list[float],
    normal: list[float],
) -> pv.DataSet:
    """Slice a volume mesh and return a mesh with cell data preserved."""
    sliced = mesh.slice(normal=normal, origin=origin)
    if sliced.n_cells == 0:
        msg = (
            f"Slice produced an empty mesh for origin={origin}, normal={normal}"
        )
        raise ValueError(msg)
    assert isinstance(sliced, pv.DataSet)
    return sliced


def _plot_scalars(mesh: pv.DataSet, field_name: str) -> np.ndarray:
    values = np.asarray(mesh.cell_data[field_name])
    if values.ndim == 2 and values.shape[1] > 1:
        return np.linalg.norm(values, axis=1)
    return values.reshape(-1)


def _field_clim(
    mesh: pv.DataSet,
    field_name: str,
) -> tuple[float, float]:
    values = _plot_scalars(mesh, field_name)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Field {field_name!r} has no finite values.")
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def set_view_mode(pl: pv.Plotter, mode: SliceMode) -> None:
    match mode:
        case SliceMode.XY:
            pl.renderer.view_xy()
        case SliceMode.XZ:
            pl.renderer.view_xz()
        case SliceMode.YZ:
            pl.renderer.view_yz()


def plot(
    slice_mode: SliceMode,
    sliced_mesh: pv.DataSet,
    plot_context: list[PlotRow],
    output_path: Path,
) -> Path:
    """Plot reference, gridfoam, and error fields for each quantity."""
    n_rows = len(plot_context)
    n_cols = max(3 + (row.err_theta_field is not None) for row in plot_context)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        shape=(n_rows, n_cols),
        window_size=[450 * n_cols, 380 * n_rows],
    )
    plotter.renderer.enable_parallel_projection()
    plotter.renderer.add_axes()
    set_view_mode(plotter, slice_mode)
    plotter.link_views()

    for row_idx, row in enumerate(plot_context):
        ref_clim = _field_clim(sliced_mesh, row.ref_field.name)
        our_clim = _field_clim(sliced_mesh, row.our_field.name)
        shared_clim = (
            min(ref_clim[0], our_clim[0]),
            max(ref_clim[1], our_clim[1]),
        )
        panels: list[tuple[int, PlotItem, tuple[float, float], str]] = [
            (0, row.ref_field, shared_clim, row.ref_field.name),
            (1, row.our_field, shared_clim, row.our_field.name),
            (
                2,
                row.err_mag_field,
                _field_clim(sliced_mesh, row.err_mag_field.name),
                row.err_mag_field.name,
            ),
        ]
        if row.err_theta_field is not None:
            panels.append(
                (
                    3,
                    row.err_theta_field,
                    _field_clim(sliced_mesh, row.err_theta_field.name),
                    row.err_theta_field.name,
                )
            )

        for col_idx, item, clim, title in panels:
            plotter.subplot(row_idx, col_idx)
            plotter.add_mesh(
                sliced_mesh,
                scalars=item.name,
                cmap=item.cmap,
                clim=clim,
                show_edges=True,
                edge_color=EDGE_COLOR,
                line_width=EDGE_LINE_WIDTH,
                edge_opacity=EDGE_OPACITY,
                scalar_bar_args={"title": title},
            )
            plotter.add_text(title, font_size=10)

    output_file = output_path.with_suffix(".png")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plotter.renderer.reset_camera(bounds=sliced_mesh.bounds)
    plotter.screenshot(str(output_file))
    plotter.close()
    return output_file


def load_openfoam_mesh(cfg: CaseConfig) -> pv.DataSet:
    vtk_dir = cfg.get_openfoam_case_dir() / "VTK"
    vtu = next(vtk_dir.rglob("internal.vtu"), None)
    if vtu is None:
        msg = f"No internal.vtu found in {vtk_dir}"
        raise FileNotFoundError(msg)
    pvmesh = pv.read(vtu)
    assert isinstance(pvmesh, pv.DataSet)
    return pvmesh


def load_gridfoam_mesh(cfg: CaseConfig) -> pv.DataSet:
    vtu_dir = cfg.get_gridfoam_case_dir() / "outputs"
    vtus = vtu_dir.glob("*[0-9][0-9][0-9][0-9].vtu")
    if not vtus:
        msg = f"No .vtu files found in {vtu_dir}"
        raise FileNotFoundError(msg)
    vtu = sorted(vtus)[-1]
    pvmesh = pv.read(vtu)
    assert isinstance(pvmesh, pv.DataSet)
    return pvmesh


def run_openfoam(of_dir: Path):
    subprocess.run(["./Allclean"], cwd=of_dir, check=True)
    subprocess.run(["./Allrun"], cwd=of_dir, check=True)


def run_gridfoam(gf_dir: Path):
    gridfoam_script = gf_dir / "run.py"
    if not gridfoam_script.is_file():
        msg = f"gridfoam script not found: {gridfoam_script}"
        raise FileNotFoundError(msg)
    subprocess.run(["uv", "run", "python", str(gridfoam_script)], check=True)


def main() -> None:
    # set MLflow tracking URI (default is ./mlruns)
    mlflow.set_tracking_uri("file:./mlruns")
    # group experiments
    mlflow.set_experiment("gridfoam vs openfoam Experiment")

    parameters = load_experiment_parameters(PARAMETERS_PATH)

    # gridfoam experiments
    for case_cfg in parameters.case:
        if case_cfg.gridfoam:
            print(f"Running gridfoam for {case_cfg.name}")
            run_gridfoam(case_cfg.get_gridfoam_case_dir())
        if case_cfg.openfoam:
            print(f"Running openfoam for {case_cfg.name}")
            run_openfoam(case_cfg.get_openfoam_case_dir())

        # load gridfoam and openfoam meshes
        gridfoam_mesh = load_gridfoam_mesh(case_cfg)
        openfoam_mesh = load_openfoam_mesh(case_cfg)

        # Fields in openfoam_mesh are prefixed with "of_", and gridfoam_mesh
        # fields are prefixed with "gf_".
        rename_fields(gridfoam_mesh, "gf_")
        rename_fields(openfoam_mesh, "of_")

        # map fields from gridfoam onto openfoam mesh
        map_fields(gridfoam_mesh, openfoam_mesh)

        # slice meshes at each slice mode
        for slice_mode in case_cfg.slice_modes:
            run_name = f"{case_cfg.name}_{slice_mode.value}"
            with mlflow.start_run(nested=True, run_name=run_name):
                mlflow.log_params(
                    {
                        "case_name": case_cfg.name,
                        "slice_mode": slice_mode.value,
                        "slice_origin": case_cfg.slice_origin,
                    }
                )

                slice_normal = case_cfg.get_slice_normal(slice_mode)
                slice_origin = case_cfg.slice_origin
                sliced_mesh = slice_mesh(
                    openfoam_mesh, slice_origin, slice_normal
                )

                # calculate error in magnitude and angle of U
                gf_U = sliced_mesh.cell_data["gf_U"]
                of_U = sliced_mesh.cell_data["of_U"]
                gf_mag_U = np.linalg.norm(gf_U, ord=2, axis=1)
                of_mag_U = np.linalg.norm(of_U, ord=2, axis=1)
                err_mag_U = np.abs(gf_mag_U - of_mag_U)
                sliced_mesh.cell_data["err_mag_U"] = err_mag_U

                denom = gf_mag_U * of_mag_U
                mask = denom > 0.0
                cos_theta = np.where(
                    mask,
                    np.sum(gf_U * of_U, axis=1) / denom,
                    0.0,
                )
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                err_theta = np.rad2deg(np.arccos(cos_theta))
                sliced_mesh.cell_data["err_theta"] = err_theta
                plot_rows_U = PlotRow(
                    ref_field=PlotItem(name="of_U", cmap=FAST_CMAP),
                    our_field=PlotItem(name="gf_U", cmap=FAST_CMAP),
                    err_mag_field=PlotItem(name="err_mag_U", cmap="magma"),
                    err_theta_field=PlotItem(name="err_theta", cmap="coolwarm"),
                )

                # calculate error in pressure
                gf_p = sliced_mesh.cell_data["gf_p"]
                of_p = sliced_mesh.cell_data["of_p"]
                err_p = np.abs(gf_p - of_p)
                sliced_mesh.cell_data["err_p"] = err_p
                plot_rows_p = PlotRow(
                    ref_field=PlotItem(name="of_p", cmap="coolwarm"),
                    our_field=PlotItem(name="gf_p", cmap="coolwarm"),
                    err_mag_field=PlotItem(name="err_p", cmap="magma"),
                    err_theta_field=None,
                )

                # plot results
                plot_context = [plot_rows_U, plot_rows_p]
                output_path = (
                    case_cfg.get_case_dir() / "plots" / slice_mode.value
                )
                output_file = plot(
                    slice_mode=slice_mode,
                    sliced_mesh=sliced_mesh,
                    plot_context=plot_context,
                    output_path=output_path,
                )

                # calculate L1, L2, and Linf errors
                err_mag_U_l1 = np.linalg.norm(err_mag_U, ord=1).item()
                err_mag_U_l2 = np.linalg.norm(err_mag_U, ord=2).item()
                err_mag_U_linf = np.linalg.norm(err_mag_U, ord=np.inf).item()
                err_theta_l1 = np.linalg.norm(err_theta, ord=1).item()
                err_theta_l2 = np.linalg.norm(err_theta, ord=2).item()
                err_theta_linf = np.linalg.norm(err_theta, ord=np.inf).item()
                err_p_l1 = np.linalg.norm(err_p, ord=1).item()
                err_p_l2 = np.linalg.norm(err_p, ord=2).item()
                err_p_linf = np.linalg.norm(err_p, ord=np.inf).item()
                metrics = {
                    "mag_U_L1": err_mag_U_l1,
                    "mag_U_L2": err_mag_U_l2,
                    "mag_U_Linf": err_mag_U_linf,
                    "theta_L1": err_theta_l1,
                    "theta_L2": err_theta_l2,
                    "theta_Linf": err_theta_linf,
                    "p_L1": err_p_l1,
                    "p_L2": err_p_l2,
                    "p_Linf": err_p_linf,
                }
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(output_file))


if __name__ == "__main__":
    main()
