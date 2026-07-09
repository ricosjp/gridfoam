"""Unit tests for diagnostics CSV collector."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.meta.config import (
    ContinuityErrorConfig,
    PostProcessingConfig,
    SolverInfoConfig,
)
from gridfoam.meta.enums import FieldRole
from gridfoam.post.diagnostics import DiagnosticsCollector
from gridfoam.solvers.base import SolveStats


@pytest.fixture
def diagnostics_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


def test_collector_writes_continuity_rows_on_interval(
    small_axis_projected_grid: AxisProjectedGrid,
    diagnostics_output_dir: Path,
) -> None:
    grid = small_axis_projected_grid
    config = PostProcessingConfig(
        continuityError=ContinuityErrorConfig(writeInterval=2),
    )
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, num_components=1)
    phi.single_data.zero_()

    collector = DiagnosticsCollector.from_config(
        grid,
        config,
        diagnostics_output_dir,
        n_steps=5,
        delta_t=1.0,
    )
    assert collector is not None
    with collector:
        for step in range(1, 6):
            collector.record_step(step, phi, {})

    csv_path = diagnostics_output_dir / "continuity_error.csv"
    with csv_path.open() as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["time", "local", "global", "cumulative"]
    assert len(rows) == 4  # header + steps 2, 4, 5
    assert rows[1][0] == "2.0"


def test_collector_writes_solver_info_with_flush(
    small_axis_projected_grid: AxisProjectedGrid,
    diagnostics_output_dir: Path,
) -> None:
    grid = small_axis_projected_grid
    config = PostProcessingConfig(
        solverInfo=SolverInfoConfig(fields=["p"], writeInterval=1),
    )
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, num_components=1)
    phi.single_data.zero_()
    stats = {
        "p": (
            SolveStats(
                solver="cg",
                initial_residual=1.0,
                final_residual=1e-4,
                iterations=3,
                converged=True,
            ),
        )
    }

    collector = DiagnosticsCollector.from_config(
        grid,
        config,
        diagnostics_output_dir,
        n_steps=1,
        delta_t=0.5,
    )
    assert collector is not None
    collector.record_step(1, phi, stats)

    csv_path = diagnostics_output_dir / "solver_info.csv"
    with csv_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "time",
        "p_solver",
        "p_initial",
        "p_final",
        "p_iters",
        "p_converged",
    ]
    assert rows[1][0] == "0.5"
    assert rows[1][1] == "cg"
    assert rows[1][5] == "1"

    collector.close()
    with csv_path.open() as f:
        assert len(list(csv.reader(f))) == 2


def test_collector_writes_vector_solver_info_per_component(
    small_axis_projected_grid: AxisProjectedGrid,
    diagnostics_output_dir: Path,
) -> None:
    grid = small_axis_projected_grid
    _u = CellField(grid, "U", role=FieldRole.LOCAL, num_components=3)
    config = PostProcessingConfig(
        solverInfo=SolverInfoConfig(fields=["U"], writeInterval=1),
    )
    phi = FaceField(grid, "phi", role=FieldRole.LOCAL, num_components=1)
    phi.single_data.zero_()
    stats = {
        "U": (
            SolveStats("bicgstab", 1.0, 1e-3, 2, True),
            SolveStats("bicgstab", 0.5, 2e-3, 3, True),
            SolveStats("bicgstab", 0.2, 3e-3, 4, False),
        )
    }

    collector = DiagnosticsCollector.from_config(
        grid,
        config,
        diagnostics_output_dir,
        n_steps=1,
        delta_t=1.0,
    )
    assert collector is not None
    collector.record_step(1, phi, stats)
    collector.close()

    csv_path = diagnostics_output_dir / "solver_info.csv"
    with csv_path.open() as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "time",
        "U_0_solver",
        "U_0_initial",
        "U_0_final",
        "U_0_iters",
        "U_0_converged",
        "U_1_solver",
        "U_1_initial",
        "U_1_final",
        "U_1_iters",
        "U_1_converged",
        "U_2_solver",
        "U_2_initial",
        "U_2_final",
        "U_2_iters",
        "U_2_converged",
    ]
    assert rows[1][4] == "2"
    assert rows[1][9] == "3"
    assert rows[1][14] == "4"
    assert rows[1][15] == "0"


def test_from_config_returns_none_when_disabled(
    small_axis_projected_grid: AxisProjectedGrid,
    diagnostics_output_dir: Path,
) -> None:
    collector = DiagnosticsCollector.from_config(
        small_axis_projected_grid,
        PostProcessingConfig(),
        diagnostics_output_dir,
        n_steps=1,
        delta_t=1.0,
    )
    assert collector is None
