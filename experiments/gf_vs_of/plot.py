import json
import pathlib
from typing import Literal

import matplotlib.colors as mcolors
import numpy as np
import pyvista as pv
from experiments.schema import CompareConfig


def _paraview_preset_to_cmap() -> mcolors.LinearSegmentedColormap:
    path = pathlib.Path(__file__).resolve().parent / "cmap_preset" / "fast.json"
    with open(path) as f:
        presets = json.load(f)
    preset = presets[0] if isinstance(presets, list) else presets
    rgb = np.array(preset["RGBPoints"]).reshape(-1, 4)

    x = rgb[:, 0]
    colors = rgb[:, 1:4]
    name = preset["Name"]

    return mcolors.LinearSegmentedColormap.from_list(
        name, list(zip(x, colors, strict=True))
    )


fast_cmap = _paraview_preset_to_cmap()
EDGE_COLOR = "black"
EDGE_LINE_WIDTH = 0.1
EDGE_OPACITY = 0.08


def set_view_mode(pl: pv.Plotter, mode: Literal["xy", "yz", "xz"]) -> None:
    match mode:
        case "xy":
            pl.renderer.view_xy()
        case "yz":
            pl.renderer.view_yz()
        case "xz":
            pl.renderer.view_xz()
        case _:
            raise ValueError(f"Invalid view mode: {mode}")


def plot(slc: pv.DataSet, cfg: CompareConfig) -> pathlib.Path:
    scalar_name = slc.field_data["scalar_name"]

    output_png = cfg.case_dir / "plots" / f"error_abs_{scalar_name}.png"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    of_min, of_max = slc.get_data_range("openfoam")
    gf_min, gf_max = slc.get_data_range("gridfoam")
    clim_common = [min(of_min, gf_min), max(of_max, gf_max)]

    pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[2400, 800])
    pl.renderer.enable_parallel_projection()
    pl.renderer.add_axes()
    set_view_mode(pl, cfg.plot.slice_mode)
    pl.link_views()

    # openfoam
    pl.subplot(0, 0)
    pl.add_mesh(
        slc,
        scalars="openfoam",
        cmap=fast_cmap,
        clim=clim_common,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=EDGE_LINE_WIDTH,
        edge_opacity=EDGE_OPACITY,
        scalar_bar_args={
            "title": f"{scalar_name} (openfoam)",
        },
    )
    pl.add_text("openfoam", font_size=11)

    # gridfoam
    pl.subplot(0, 1)
    pl.add_mesh(
        slc,
        scalars="gridfoam",
        cmap=fast_cmap,
        clim=clim_common,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=EDGE_LINE_WIDTH,
        edge_opacity=EDGE_OPACITY,
        scalar_bar_args={
            "title": f"{scalar_name} (gridfoam)",
        },
    )
    pl.add_text("gridfoam", font_size=11)

    # error_abs
    pl.subplot(0, 2)
    pl.add_mesh(
        slc.copy(),
        scalars="error_abs",
        cmap="magma",
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=EDGE_LINE_WIDTH,
        edge_opacity=EDGE_OPACITY,
        scalar_bar_args={"title": "error_abs"},
    )
    pl.add_text("error_abs", font_size=11)

    pl.renderer.reset_camera()
    pl.screenshot(str(output_png))
    pl.close()
    return output_png
