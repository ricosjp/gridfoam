from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.factory import apply_boundary_condition_configs
from gridfoam.core.field import CellField
from gridfoam.core.grid.factory import create_grid as create_grid_from_config
from gridfoam.initialization import initialize_from_dirichlet_patch
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import GridfoamConfig
from gridfoam.meta.enums import DomainBoundaryPatch, FieldRole
from gridfoam.models.turbulence.laminar import Laminar
from gridfoam.post.forces import ForceCoeffs, ForceEvaluator

LOG_FILE_NAME = "low_re_sphere.log"

logger = logging.getLogger("gridfoam.examples.low_re_sphere")


def _read_raw_config(config_path: Path) -> dict[str, Any]:
    with config_path.open() as f:
        return yaml.safe_load(f)


def configure_run_logger(log_file: Path) -> None:
    formatter = logging.Formatter("%(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def write_force_coeffs(path: Path, coeffs: list[ForceCoeffs]) -> None:
    fieldnames = [
        "time",
        "Cd",
        "Cs",
        "Cl",
        "CmRoll",
        "CmPitch",
        "CmYaw",
        "Cd_f",
        "Cd_r",
        "Cs_f",
        "Cs_r",
        "Cl_f",
        "Cl_r",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for co in coeffs:
            writer.writerow(
                {
                    "time": co.time,
                    "Cd": co.Cd.item(),
                    "Cs": co.Cs.item(),
                    "Cl": co.Cl.item(),
                    "CmRoll": co.CmRoll.item(),
                    "CmPitch": co.CmPitch.item(),
                    "CmYaw": co.CmYaw.item(),
                    "Cd_f": co.Cd_f.item(),
                    "Cd_r": co.Cd_r.item(),
                    "Cs_f": co.Cs_f.item(),
                    "Cs_r": co.Cs_r.item(),
                    "Cl_f": co.Cl_f.item(),
                    "Cl_r": co.Cl_r.item(),
                }
            )


def _solve_single_case(
    config_path: Path, re_value: float, output_dir: Path
) -> float:
    raw_yaml = _read_raw_config(config_path)
    sim = raw_yaml["simulator"]
    force_coeff = sim["forceCoeff"]
    u_ref = float(force_coeff["magU_ref"])
    l_ref = float(force_coeff["L_ref"])
    nu = u_ref * l_ref / re_value
    sim["control"]["output"]["output_dir"] = str(output_dir)

    config = GridfoamConfig.model_validate(raw_yaml)
    grid = create_grid_from_config(config)
    U = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    p = CellField(grid, "p", role=FieldRole.LOCAL, num_components=1)

    boundary_conditions = grid.sim_config.boundaryConditions
    if boundary_conditions is None:
        raise ValueError("boundaryConditions is required for this example.")
    apply_boundary_condition_configs(U, boundary_conditions["U"])
    apply_boundary_condition_configs(p, boundary_conditions["p"])
    initialize_from_dirichlet_patch(
        U, boundary_conditions["U"], DomainBoundaryPatch.X_MINUS
    )

    turbulence = Laminar(grid=grid, nu=nu)

    force_config = grid.sim_config.forceCoeff
    if force_config is None:
        raise ValueError("forceCoeff is required for this example.")
    force_evaluator = ForceEvaluator(force_config)

    algo = SIMPLE(grid=grid, U=U, p=p, turbulence=turbulence)

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(output_dir / LOG_FILE_NAME)
    write_interval = grid.sim_config.control.writeInterval
    ugrid = to_unstructured_grid(grid)

    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )

    for step in range(1, n_steps + 1):
        algo.step()
        if step % write_interval == 0 or step == n_steps:
            force_evaluator.evaluate(
                grid,
                time=float(step),
                p=p,
                U=U,
                turbulence=turbulence,
            )
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"low_re_sphere_{step:04d}.vtu"),
                ugrid=ugrid,
            )
            max_u = torch.linalg.vector_norm(U.data, ord=2, dim=1).max().item()
            cd = force_evaluator.history[-1].Cd.item()
            logger.info(
                "step=%4d Re=%.3g max|U|=%.4e Cd=%.6e",
                step,
                re_value,
                max_u,
                cd,
            )

    write_force_coeffs(output_dir / "force_coeffs.csv", force_evaluator.history)
    grid.surface_mesh.save(
        output_dir / "surface_mesh.vtu",
        overwrite_features=True,
        overwrite_file=True,
    )
    return force_evaluator.history[-1].Cd.item()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-Re sphere cases.")
    parser.add_argument(
        "--re",
        required=True,
        type=float,
        nargs="+",
        default=[1.0],
        help="One or more Reynolds numbers.",
    )
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(6)
    args = _parse_args()
    config_path = Path(__file__).resolve().parent / "data" / "config.yml"
    base_output = Path(__file__).resolve().parent / "outputs"

    summary_rows: list[dict[str, float]] = []
    for re_value in args.re:
        re_dir = base_output / f"re_{re_value:g}"
        cd_value = _solve_single_case(config_path, re_value, re_dir)
        summary_rows.append({"Re": re_value, "Cd": cd_value})

    summary_path = base_output / "sweep_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Re", "Cd"])
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
