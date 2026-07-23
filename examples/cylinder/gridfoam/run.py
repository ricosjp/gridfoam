"""Transient (PIMPLE) flow around a circular cylinder at Re=100.

Runs the gridfoam PIMPLE solver and records the time history of the drag and
lift coefficients so that the von Karman vortex shedding can be compared with
the matched OpenFOAM pimpleFoam case under ``examples/cylinder/openfoam``.
"""

import logging
import sys
from pathlib import Path

import torch

from gridfoam.io.vtu import save_export_fields_as_vtu
from gridfoam.post.forces.evaluator import ForceEvaluator
from gridfoam.runner import manual_step

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def configure_run_logger(log_file: Path) -> None:
    formatter = logging.Formatter("%(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main() -> None:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logger(output_dir / "cylinder.log")

    config_path = ROOT / "data" / "config.yaml"

    force_evaluator: ForceEvaluator | None = None
    for step_data in manual_step(config_path):
        algorithm = step_data.algorithm
        algorithm.step()

        grid = algorithm.grid
        if force_evaluator is None:
            force_evaluator = ForceEvaluator(grid)

        control = step_data.control
        delta_t = control.deltaT
        end_time = control.endTime
        n_steps = int(end_time / delta_t)
        time = step_data.step * delta_t

        # Sample the force coefficients every step to resolve the shedding.
        coeffs = force_evaluator.evaluate(
            grid, time=time, turbulence=algorithm.turbulence
        )

        if step_data.step % control.writeInterval == 0 or (
            step_data.step == n_steps
        ):
            base_name = control.output.base_name
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"{base_name}_{step_data.step:04d}.vtu"),
                ugrid=step_data.ugrid,
            )
            u_field = grid.get_cellfield("U")
            assert u_field is not None
            max_u = (
                torch.linalg.vector_norm(u_field.data, ord=2, dim=1)
                .max()
                .item()
            )
            logger.info(
                "step=%5d t=%7.2f max|U|=%.4e Cd=%.5f Cl=%.5f",
                step_data.step,
                time,
                max_u,
                coeffs.Cd.item(),
                coeffs.Cl.item(),
            )

    if force_evaluator is not None:
        force_evaluator.write_csv(output_dir / "coefficients.csv")
        force_evaluator.save_surface_mesh(output_dir / "surface_mesh.vtu")


if __name__ == "__main__":
    main()
