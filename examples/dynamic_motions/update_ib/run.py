"""PISO flow around a translating square prism using ``update_ib``.

The background mesh is fixed. A refinement corridor covers the short
+x travel so IBM data stays at ``target_level``. Each step poses the
body, sets the immersed Dirichlet to the body velocity, refreshes
``phi``, then solves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from gridfoam.algorithms.factory import create_algorithm
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.config import ManualAlgorithm
from gridfoam.runner import manual_run

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.dynamic_motions._common import (  # noqa: E402
    configure_run_logger,
    immersed_wall_velocity,
    refresh_flux,
    set_immersed_wall_velocity,
    write_square_prism_stl,
    write_step_outputs,
)

# Rest-position prism (xmin, xmax, ymin, ymax, zmin, zmax).
BODY_BOUNDS = (0.40, 0.60, 0.40, 0.60, 0.00, 0.10)


def main() -> None:
    logger = configure_run_logger("gridfoam.examples.update_ib")
    case_dir = Path(__file__).resolve().parent
    write_square_prism_stl(case_dir / "data" / "cube.stl", bounds=BODY_BOUNDS)

    grid = manual_run(case_dir / "data" / "config.yaml")
    assert isinstance(grid, AxisProjectedGrid)
    assert grid.is_dynamic

    if isinstance(grid.sim_config.fvSolution.algorithm, ManualAlgorithm):
        raise ValueError("Manual algorithm has no step sequence")

    algorithm = create_algorithm(grid)
    body_velocity = immersed_wall_velocity(grid)
    control = grid.sim_config.control
    n_steps = int(control.endTime / control.deltaT)
    output_dir = Path(control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "update_ib demo: n_cells=%d immersed=%d n_steps=%d U_wall=%s",
        grid.num_cells,
        grid.num_immersed_faces,
        n_steps,
        body_velocity,
    )

    n_cells0 = grid.num_cells
    for step in range(1, n_steps + 1):
        time = step * control.deltaT
        translation = [v * time for v in body_velocity]
        grid.update_ib(translation=translation)
        set_immersed_wall_velocity(grid, body_velocity)
        refresh_flux(grid)

        algorithm.step()

        if step % control.writeInterval == 0 or step == n_steps:
            write_step_outputs(
                grid,
                output_dir=output_dir,
                base_name=control.output.base_name,
                step=step,
            )
            u_field = grid.get_cellfield("U")
            assert u_field is not None
            max_u = (
                torch.linalg.vector_norm(u_field.data, ord=2, dim=1)
                .max()
                .item()
            )
            logger.info(
                "step=%4d t=%.3f tx=%.3f cells=%d immersed=%d max|U|=%.4e",
                step,
                time,
                translation[0],
                grid.num_cells,
                grid.num_immersed_faces,
                max_u,
            )

    if grid.num_cells != n_cells0:
        raise RuntimeError("update_ib must keep background topology fixed")
    logger.info("wrote outputs under %s", output_dir)


if __name__ == "__main__":
    main()
