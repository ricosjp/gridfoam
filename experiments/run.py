import argparse
import gc
import pathlib

import mlflow
import pyvista as pv
import yaml

from experiments.compare import compare_on_slice
from experiments.mesh_io import (
    load_gridfoam_mesh,
    load_openfoam_mesh,
)
from experiments.plot import plot
from experiments.run_cases import run_gridfoam, run_openfoam
from experiments.schema import CompareConfig


def load_config(path: pathlib.Path) -> CompareConfig:
    """
    Load CompareConfig from a YAML file and validate with Pydantic.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    CompareConfig
        Parsed and validated configuration.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return CompareConfig.model_validate(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a gridfoam experiment."
            "All run parameters are read from a YAML configuration file."
        )
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    openfoam_timing = run_openfoam(cfg)
    gridfoam_timing = run_gridfoam(cfg)

    of_mesh = load_openfoam_mesh(cfg)
    gf_mesh = load_gridfoam_mesh(cfg)

    slc = compare_on_slice(of_mesh, gf_mesh, cfg)

    # plot slice
    output_png = plot(slc, cfg)

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "field_name": cfg.field_name,
                "field_kind": cfg.field_kind,
                "slice_mode": cfg.plot.slice_mode,
                "slice_origin": cfg.plot.slice_origin,
            }
        )
        metrics = {
            "Linf": slc.field_data["Linf"].item(),
            "L2": slc.field_data["L2"].item(),
            "L1": slc.field_data["L1"].item(),
        }
        if openfoam_timing is not None:
            metrics.update(
                {
                    "openfoam_wall_time_s": openfoam_timing.wall_time_s,
                    "openfoam_cpu_user_s": openfoam_timing.cpu_user_s,
                    "openfoam_cpu_system_s": openfoam_timing.cpu_system_s,
                    "openfoam_cpu_total_s": openfoam_timing.cpu_total_s,
                }
            )
        if gridfoam_timing is not None:
            metrics.update(
                {
                    "gridfoam_wall_time_s": gridfoam_timing.wall_time_s,
                    "gridfoam_cpu_user_s": gridfoam_timing.cpu_user_s,
                    "gridfoam_cpu_system_s": gridfoam_timing.cpu_system_s,
                    "gridfoam_cpu_total_s": gridfoam_timing.cpu_total_s,
                }
            )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_png))

    del of_mesh, gf_mesh, slc
    pv.close_all()
    gc.collect()
