import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyvista as pv
import yaml

from gridfoam.algorithms.base import AlgorithmBase
from gridfoam.algorithms.factory import create_algorithm
from gridfoam.core.grid.base import IGridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.io.vtu import save_export_fields_as_vtu, to_unstructured_grid
from gridfoam.meta.config import ControlConfig, GridfoamConfig, ManualAlgorithm
from gridfoam.post.diagnostics import DiagnosticsCollector
from gridfoam.post.forces import ForceEvaluator
from gridfoam.pre.potential_flow import PotentialFlow

logger = logging.getLogger(__name__)


@dataclass
class StepData:
    """
    Data for a single step of the simulation.
    """

    step: int
    """
    current step number.
    """
    control: ControlConfig
    """
    Control configuration.
    """
    algorithm: AlgorithmBase
    """
    Algorithm.
    """
    ugrid: pv.UnstructuredGrid
    """
    Unstructured grid.
    """


def manual_run(config_path: Path) -> IGridBase:
    with open(config_path) as f:
        raw_yaml = yaml.safe_load(f)
    config = GridfoamConfig.model_validate(raw_yaml)
    grid = create_grid(config)
    return grid


def manual_step(config_path: Path) -> Iterator[StepData]:
    grid = manual_run(config_path)

    phase = None
    # Create the algorithm
    algorithm_config = grid.sim_config.fvSolution.algorithm
    if isinstance(algorithm_config, ManualAlgorithm):
        raise ValueError("Manual algorithm has no step sequence")

    if grid.sim_config.fvSolution.potentialFlow is not None:
        # OpenFOAM potentialFoam-style initialization before the main solver
        PotentialFlow(grid, phase).solve()

    algorithm = create_algorithm(grid, phase)

    output_dir = Path(grid.sim_config.control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ugrid = to_unstructured_grid(grid)

    n_steps = int(
        grid.sim_config.control.endTime / grid.sim_config.control.deltaT
    )
    diagnostics = DiagnosticsCollector.from_config(
        grid,
        grid.sim_config.post_processing,
        output_dir,
        n_steps=n_steps,
        delta_t=grid.sim_config.control.deltaT,
        phase=phase,
    )
    if diagnostics is not None:
        algorithm.attach_diagnostics(diagnostics)

    try:
        for step in range(1, n_steps + 1):
            yield StepData(
                step=step,
                control=grid.sim_config.control,
                algorithm=algorithm,
                ugrid=ugrid,
            )

        post_processing = grid.sim_config.post_processing
        if post_processing is not None:
            if post_processing.forceCoeff is not None:
                force_evaluator = ForceEvaluator(grid, phase=phase)
                force_evaluator.evaluate(
                    grid,
                    time=float(grid.sim_config.control.endTime),
                    turbulence=algorithm.turbulence,
                )
                force_evaluator.write_csv(output_dir / "coefficients.csv")
                force_evaluator.save_surface_mesh(
                    output_dir / "surface_mesh.vtu"
                )
    finally:
        if diagnostics is not None:
            diagnostics.close()


def _algorithm_converged(algorithm: AlgorithmBase) -> bool:
    """
    Return whether an algorithm reports pseudo-time / outer-loop convergence.

    Algorithms without ``has_converged()`` always return ``False``.
    """
    has_converged = getattr(algorithm, "has_converged", None)
    if callable(has_converged):
        return bool(has_converged())
    return False


def all_run(config_path: Path) -> None:
    for step_data in manual_step(config_path):
        step_data.algorithm.step()
        if _algorithm_converged(step_data.algorithm):
            logger.info("Simulation converged at step=%d", step_data.step)
            break
        write_interval = step_data.control.writeInterval
        end_time = step_data.control.endTime
        deltaT = step_data.control.deltaT
        n_steps = int(end_time / deltaT)
        if step_data.step % write_interval == 0 or step_data.step == n_steps:
            output_dir = Path(step_data.control.output.output_dir)
            base_name = step_data.control.output.base_name
            save_export_fields_as_vtu(
                step_data.algorithm.grid,
                str(output_dir / f"{base_name}_{step_data.step:04d}.vtu"),
                ugrid=step_data.ugrid,
            )
            logger.info("step=%4d", step_data.step)
