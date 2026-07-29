"""Plot sequential gridfoam / OpenFOAM / diff comparisons for the cylinder case.

Creates three-column XY-slice images of velocity magnitude with annotations
matching the style of ``gridfoam/outputs/anim`` and ``openfoam/anim``:

- top-left: ``cylinder-PIMPLE-{gridfoam|openfoam|diff}``
- bottom-left: ``Time: <seconds>``
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent
GF_OUTPUT_DIR = ROOT / "gridfoam" / "outputs"
OF_VTK_DIR = ROOT / "openfoam" / "VTK"
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "anim"
FAST_CMAP_PATH = ROOT.parents[1] / "experiments" / "utils" / "fast.json"

DELTA_T = 0.05
SLICE_ORIGIN = (0.0, 0.0, 0.0)
SLICE_NORMAL = (0.0, 0.0, 1.0)
BACKGROUND = "#525252"
EDGE_COLOR = "black"
EDGE_LINE_WIDTH = 0.1
EDGE_OPACITY = 0.08
WINDOW_SIZE = (1640 * 3, 849)

GF_VTU_PATTERN = re.compile(r"^cylinder_(\d+)\.vtu$")
OF_DIR_PATTERN = re.compile(r"^openfoam_(\d+)$")


def _paraview_preset_to_cmap() -> mcolors.LinearSegmentedColormap:
    with FAST_CMAP_PATH.open() as f:
        presets = json.load(f)
    preset = presets[0] if isinstance(presets, list) else presets
    rgb = np.array(preset["RGBPoints"]).reshape(-1, 4)
    return mcolors.LinearSegmentedColormap.from_list(
        preset["Name"],
        list(zip(rgb[:, 0], rgb[:, 1:4], strict=True)),
    )


FAST_CMAP = _paraview_preset_to_cmap()


@dataclass(frozen=True)
class FramePaths:
    step: int
    time: float
    gridfoam: Path
    openfoam: Path


@dataclass
class FrameData:
    step: int
    time: float
    gridfoam: pv.DataSet
    openfoam: pv.DataSet
    diff: pv.DataSet
    mag_clim: tuple[float, float]
    diff_clim: tuple[float, float]


def _discover_frames() -> list[FramePaths]:
    gf_by_step: dict[int, Path] = {}
    for path in GF_OUTPUT_DIR.glob("cylinder_*.vtu"):
        match = GF_VTU_PATTERN.match(path.name)
        if match is None:
            continue
        gf_by_step[int(match.group(1))] = path

    of_by_step: dict[int, Path] = {}
    for path in OF_VTK_DIR.iterdir():
        if not path.is_dir():
            continue
        match = OF_DIR_PATTERN.match(path.name)
        if match is None:
            continue
        vtu = path / "internal.vtu"
        if vtu.is_file():
            of_by_step[int(match.group(1))] = vtu

    common_steps = sorted(set(gf_by_step) & set(of_by_step))
    if not common_steps:
        msg = (
            "No matching timesteps found between "
            f"{GF_OUTPUT_DIR} and {OF_VTK_DIR}"
        )
        raise FileNotFoundError(msg)

    return [
        FramePaths(
            step=step,
            time=step * DELTA_T,
            gridfoam=gf_by_step[step],
            openfoam=of_by_step[step],
        )
        for step in common_steps
    ]


def _mag_u(mesh: pv.DataSet, field: str = "U") -> np.ndarray:
    values = np.asarray(mesh.cell_data[field])
    if values.ndim == 2 and values.shape[1] > 1:
        return np.linalg.norm(values, axis=1)
    return values.reshape(-1)


def _finite_clim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Field has no finite values.")
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def _slice_xy(mesh: pv.DataSet) -> pv.DataSet:
    sliced = mesh.slice(normal=SLICE_NORMAL, origin=SLICE_ORIGIN)
    if sliced.n_cells == 0:
        msg = (
            "Slice produced an empty mesh for "
            f"origin={SLICE_ORIGIN}, normal={SLICE_NORMAL}"
        )
        raise ValueError(msg)
    assert isinstance(sliced, pv.DataSet)
    return sliced


def _prepare_frame(frame: FramePaths) -> FrameData:
    gf_volume = pv.read(frame.gridfoam)
    of_volume = pv.read(frame.openfoam)
    assert isinstance(gf_volume, pv.DataSet)
    assert isinstance(of_volume, pv.DataSet)

    gf_slice = _slice_xy(gf_volume)
    of_slice = _slice_xy(of_volume)

    gf_slice.cell_data["mag_U"] = _mag_u(gf_slice)
    of_slice.cell_data["mag_U"] = _mag_u(of_slice)

    probed = of_slice.cell_centers().sample(gf_volume)
    gf_u_on_of = np.asarray(probed.point_data["U"])
    of_u = np.asarray(of_slice.cell_data["U"])
    gf_mag = np.linalg.norm(gf_u_on_of, axis=1)
    of_mag = np.linalg.norm(of_u, axis=1)
    err_mag = np.abs(gf_mag - of_mag)

    diff_slice = of_slice.copy(deep=True)
    diff_slice.cell_data.clear()
    diff_slice.point_data.clear()
    diff_slice.cell_data["mag_U_diff"] = err_mag

    mag_clim = _finite_clim(
        np.concatenate(
            [
                np.asarray(gf_slice.cell_data["mag_U"]),
                np.asarray(of_slice.cell_data["mag_U"]),
            ]
        )
    )
    return FrameData(
        step=frame.step,
        time=frame.time,
        gridfoam=gf_slice,
        openfoam=of_slice,
        diff=diff_slice,
        mag_clim=mag_clim,
        diff_clim=_finite_clim(err_mag),
    )


def _add_panel(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    *,
    scalars: str,
    clim: tuple[float, float],
    cmap: str | mcolors.LinearSegmentedColormap,
    title: str,
    time: float,
    colorbar_title: str,
) -> None:
    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=EDGE_LINE_WIDTH,
        edge_opacity=EDGE_OPACITY,
        scalar_bar_args={
            "title": colorbar_title,
            "title_font_size": 14,
            "label_font_size": 12,
            "n_labels": 5,
            "fmt": "%.1e",
            "color": "white",
            "vertical": True,
            "position_x": 0.85,
            "position_y": 0.2,
            "width": 0.08,
            "height": 0.6,
        },
    )
    plotter.add_axes(viewport=(0.0, 0.0, 0.2, 0.2))
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.add_text(
        title,
        position="upper_left",
        font_size=12,
        color="white",
        shadow=True,
    )
    plotter.add_text(
        f"Time: {time:.6f}",
        position="lower_left",
        font_size=12,
        color="white",
        shadow=True,
    )


def plot_frame(
    frame: FrameData,
    output_path: Path,
    *,
    mag_clim: tuple[float, float] | None = None,
    diff_clim: tuple[float, float] | None = None,
) -> Path:
    mag_clim = mag_clim or frame.mag_clim
    diff_clim = diff_clim or frame.diff_clim

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        shape=(1, 3),
        window_size=list(WINDOW_SIZE),
    )
    plotter.set_background(BACKGROUND)

    panels = (
        (
            0,
            frame.gridfoam,
            "mag_U",
            mag_clim,
            FAST_CMAP,
            "cylinder-PIMPLE-gridfoam",
            "U Magnitude",
        ),
        (
            1,
            frame.openfoam,
            "mag_U",
            mag_clim,
            FAST_CMAP,
            "cylinder-PIMPLE-openfoam",
            "U Magnitude",
        ),
        (
            2,
            frame.diff,
            "mag_U_diff",
            diff_clim,
            "magma",
            "cylinder-PIMPLE-diff",
            "|ΔU| Magnitude",
        ),
    )
    for col, mesh, scalars, clim, cmap, title, bar_title in panels:
        plotter.subplot(0, col)
        _add_panel(
            plotter,
            mesh,
            scalars=scalars,
            clim=clim,
            cmap=cmap,
            title=title,
            time=frame.time,
            colorbar_title=bar_title,
        )

    plotter.link_views()
    plotter.renderer.reset_camera(bounds=frame.openfoam.bounds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path))
    plotter.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for PNG frames (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of write steps to plot (e.g. 100 500 3000).",
    )
    parser.add_argument(
        "--global-clim",
        action="store_true",
        help="Use min/max over all selected frames for stable animation scales.",
    )
    args = parser.parse_args()

    frames = _discover_frames()
    if args.steps is not None:
        wanted = set(args.steps)
        frames = [frame for frame in frames if frame.step in wanted]
        missing = wanted - {frame.step for frame in frames}
        if missing:
            msg = f"Requested steps not found in both solvers: {sorted(missing)}"
            raise FileNotFoundError(msg)

    prepared: list[FrameData] = []
    for index, frame in enumerate(frames):
        print(
            f"[{index + 1}/{len(frames)}] preparing step={frame.step} "
            f"t={frame.time:.6f}"
        )
        prepared.append(_prepare_frame(frame))

    mag_clim: tuple[float, float] | None = None
    diff_clim: tuple[float, float] | None = None
    if args.global_clim and prepared:
        mag_clim = (
            min(item.mag_clim[0] for item in prepared),
            max(item.mag_clim[1] for item in prepared),
        )
        diff_clim = (
            min(item.diff_clim[0] for item in prepared),
            max(item.diff_clim[1] for item in prepared),
        )
        print(f"global mag_U clim={mag_clim}")
        print(f"global |ΔU| clim={diff_clim}")

    for index, frame_data in enumerate(prepared):
        output_path = args.output_dir / f"xy.{index:04d}.png"
        print(f"[{index + 1}/{len(prepared)}] writing {output_path}")
        plot_frame(
            frame_data,
            output_path,
            mag_clim=mag_clim,
            diff_clim=diff_clim,
        )


if __name__ == "__main__":
    main()
