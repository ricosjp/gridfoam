"""PISO flow around a translating square prism using ``remesh``.

The body travels farther than the initial surface-refinement band.
``remesh`` rebuilds AMR around the current pose every step. ``U``, ``p``,
and ``phi`` are nearest-neighbour mapped from the previous mesh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from gridfoam.algorithms.factory import create_algorithm
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.remap import capture_volume_fields, map_volume_fields
from gridfoam.meta.config import ManualAlgorithm
from gridfoam.runner import manual_run

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.dynamic_motions._common import (  # noqa: E402
    configure_run_logger,
    refresh_flux,
    write_square_prism_stl,
    write_step_outputs,
)

# Rest-position prism (x, y, z) and constant +x body speed.
BODY_BOUNDS = (0.40, 0.60, 0.40, 0.60, 0.00, 0.10)
BODY_SPEED = 1.5


def main() -> None:
    logger = configure_run_logger("gridfoam.examples.remesh")
    case_dir = Path(__file__).resolve().parent
    write_square_prism_stl(case_dir / "data" / "cube.stl", bounds=BODY_BOUNDS)

    grid = manual_run(case_dir / "data" / "config.yaml")
    assert isinstance(grid, AxisProjectedGrid)
    assert grid.is_dynamic

    if isinstance(grid.sim_config.fvSolution.algorithm, ManualAlgorithm):
        raise ValueError("Manual algorithm has no step sequence")

    algorithm = create_algorithm(grid)
    control = grid.sim_config.control
    n_steps = int(control.endTime / control.deltaT)
    output_dir = Path(control.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "remesh demo: n_cells=%d immersed=%d n_steps=%d speed=%.3f",
        grid.num_cells,
        grid.num_immersed_faces,
        n_steps,
        BODY_SPEED,
    )

    for step in range(1, n_steps + 1):
        algorithm.step()

        time = step * control.deltaT
        translation = [BODY_SPEED * time, 0.0, 0.0]
        snapshot = capture_volume_fields(grid)
        grid.remesh(
            translation=translation,
            warn_outside_refinement=False,
        )
        map_volume_fields(grid, snapshot)
        algorithm = create_algorithm(grid)
        refresh_flux(grid, update_internal=False)

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

    logger.info("wrote outputs under %s", output_dir)


if __name__ == "__main__":
    main()
