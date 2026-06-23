import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from pydantic import BaseModel, ConfigDict

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_PLOT_DIR = DEFAULT_INPUT_DIR / "plots"
CASE_NAMES = ("linear", "quadratic_z", "radial")
GRAD_SCHEMES = ("linear", "leastsquare")
COMPONENTS = ("x", "y", "z")
Y_SLICE = 0.0
SLICE_NORMAL = (0.0, 1.0, 0.0)
EDGE_COLOR = "black"
EDGE_LINE_WIDTH = 0.1
EDGE_OPACITY = 0.08


class PlotItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    cmap: str


class PlotRow(BaseModel):
    numerical: PlotItem
    exact: PlotItem
    diff: PlotItem


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


def prepare_y_slice(mesh: pv.DataSet, y_slice: float) -> pv.DataSet:
    """Slice at y = y_slice and add per-component gradient fields."""
    sliced = slice_mesh(mesh, [0.0, y_slice, 0.0], list(SLICE_NORMAL))

    grad_num = np.asarray(sliced.cell_data["grad_p"], dtype=np.float64)
    grad_exact = np.asarray(sliced.cell_data["grad_p_exact"], dtype=np.float64)

    for idx, comp in enumerate(COMPONENTS):
        sliced.cell_data[f"grad{comp}_num"] = grad_num[:, idx]
        sliced.cell_data[f"grad{comp}_exact"] = grad_exact[:, idx]
        sliced.cell_data[f"grad{comp}_diff"] = np.abs(
            grad_num[:, idx] - grad_exact[:, idx]
        )

    return sliced


def _plot_scalars(mesh: pv.DataSet, field_name: str) -> np.ndarray:
    values = np.asarray(mesh.cell_data[field_name])
    if values.ndim == 2 and values.shape[1] > 1:
        return np.linalg.norm(values, axis=1)
    return values.reshape(-1)


def _field_clim(
    mesh: pv.DataSet,
    field_name: str,
    *,
    nonnegative: bool = False,
) -> tuple[float, float]:
    values = _plot_scalars(mesh, field_name)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Field {field_name!r} has no finite values.")

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        if nonnegative:
            pad = 1.0 if np.isclose(vmax, 0.0) else abs(vmax) * 0.05
            return 0.0, vmax + pad
        pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    if nonnegative:
        return 0.0, vmax
    return vmin, vmax


def _grad_plot_rows() -> list[PlotRow]:
    rows: list[PlotRow] = []
    for comp in COMPONENTS:
        rows.append(
            PlotRow(
                numerical=PlotItem(
                    name=f"grad{comp}_num",
                    cmap="coolwarm",
                ),
                exact=PlotItem(
                    name=f"grad{comp}_exact",
                    cmap="coolwarm",
                ),
                diff=PlotItem(
                    name=f"grad{comp}_diff",
                    cmap="magma",
                ),
            )
        )
    return rows


def plot(
    sliced_mesh: pv.DataSet,
    plot_rows: list[PlotRow],
    output_path: Path,
) -> Path:
    """Plot p and gradient comparison rows on an x-z slice."""
    n_rows = 1 + len(plot_rows)
    n_cols = 3

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        shape=(n_rows, n_cols),
        window_size=[450 * n_cols, 380 * n_rows],
    )
    plotter.renderer.enable_parallel_projection()
    plotter.renderer.add_axes()
    plotter.renderer.view_xz()
    plotter.link_views()

    plotter.subplot(0, 0)
    plotter.add_mesh(
        sliced_mesh,
        scalars="p",
        cmap="coolwarm",
        clim=_field_clim(sliced_mesh, "p"),
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=EDGE_LINE_WIDTH,
        edge_opacity=EDGE_OPACITY,
        scalar_bar_args={"title": "p"},
    )
    plotter.add_text("p", font_size=10)

    for row_idx, row in enumerate(plot_rows, start=1):
        num_clim = _field_clim(sliced_mesh, row.numerical.name)
        exact_clim = _field_clim(sliced_mesh, row.exact.name)
        shared_clim = (
            min(num_clim[0], exact_clim[0]),
            max(num_clim[1], exact_clim[1]),
        )
        comp = COMPONENTS[row_idx - 1]
        panels: list[tuple[int, PlotItem, tuple[float, float], str]] = [
            (0, row.numerical, shared_clim, f"grad{comp} p"),
            (1, row.exact, shared_clim, "exact"),
            (
                2,
                row.diff,
                _field_clim(sliced_mesh, row.diff.name, nonnegative=True),
                "diff",
            ),
        ]

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


def plot_case(
    vtu_path: Path,
    output_path: Path,
    *,
    y_slice: float = Y_SLICE,
) -> Path:
    """Create a comparison figure for one VTU file on the y=0 slice."""
    volume = pv.read(vtu_path)
    if not isinstance(volume, pv.DataSet):
        raise TypeError(f"Expected a DataSet, got {type(volume)!r}")
    sliced_mesh = prepare_y_slice(volume, y_slice)
    return plot(sliced_mesh, _grad_plot_rows(), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot y-slice comparisons for gradient verification VTU files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing VTU files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help=f"Directory for PNG figures. Default: {DEFAULT_PLOT_DIR}",
    )
    parser.add_argument(
        "--y-slice",
        type=float,
        default=Y_SLICE,
        help="Y coordinate of the slice plane. Default: 0.0",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for case_name in CASE_NAMES:
        for scheme in GRAD_SCHEMES:
            vtu_path = args.input_dir / f"{case_name}_{scheme}.vtu"
            if not vtu_path.exists():
                raise FileNotFoundError(f"VTU file not found: {vtu_path}")

            output_path = args.output_dir / f"{case_name}_{scheme}.png"
            output_file = plot_case(
                vtu_path,
                output_path,
                y_slice=args.y_slice,
            )
            print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
