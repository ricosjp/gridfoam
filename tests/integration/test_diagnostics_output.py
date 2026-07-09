"""Integration tests for diagnostics CSV output."""

from __future__ import annotations

import csv
from pathlib import Path

from tests.helpers import simple_convergence_config

from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import (
    ContinuityErrorConfig,
    GridfoamConfig,
    PostProcessingConfig,
    SolverInfoConfig,
)
from gridfoam.post.diagnostics import DiagnosticsCollector


def _config_with_diagnostics(tmp_path: Path) -> GridfoamConfig:
    base = simple_convergence_config(tmp_path)
    return base.model_copy(
        update={
            "simulator": base.simulator.model_copy(
                update={
                    "control": base.simulator.control.model_copy(
                        update={"endTime": 3.0}
                    ),
                    "post_processing": PostProcessingConfig(
                        continuityError=ContinuityErrorConfig(),
                        solverInfo=SolverInfoConfig(fields=["U", "p"]),
                    ),
                }
            )
        }
    )


def test_simple_run_writes_diagnostics_csv(tmp_path: Path) -> None:
    config = _config_with_diagnostics(tmp_path)
    grid = create_grid(config)
    algo = SIMPLE(grid)

    output_dir = Path(config.simulator.control.output.output_dir)
    diagnostics = DiagnosticsCollector.from_config(
        grid,
        config.simulator.post_processing,
        output_dir,
        n_steps=3,
        delta_t=config.simulator.control.deltaT,
    )
    assert diagnostics is not None
    algo.attach_diagnostics(diagnostics)

    try:
        for _ in range(3):
            algo.step()
    finally:
        diagnostics.close()

    continuity_path = output_dir / "continuity_error.csv"
    solver_info_path = output_dir / "solver_info.csv"
    assert continuity_path.is_file()
    assert solver_info_path.is_file()

    with continuity_path.open() as f:
        continuity_rows = list(csv.reader(f))
    with solver_info_path.open() as f:
        solver_rows = list(csv.reader(f))

    assert len(continuity_rows) == 4
    assert continuity_rows[0] == ["time", "local", "global", "cumulative"]
    assert len(solver_rows) == 4
    assert solver_rows[0][0] == "time"
    assert "U_0_solver" in solver_rows[0]
    assert "p_solver" in solver_rows[0]
