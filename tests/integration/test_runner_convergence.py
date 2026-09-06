"""Steady convergence ends a run; PIMPLE convergence ends only an outer loop."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import pyvista as pv
from tests.helpers import channel_flow_config, simple_convergence_config

from gridfoam import runner
from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.algorithms.simple import SIMPLE
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import PIMPLEAlgorithm, ResidualControlEntry
from gridfoam.meta.enums import AlgorithmType


@pytest.mark.parametrize("transient", [False, True])
def test_runner_stops_only_for_simulation_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transient: bool
):
    if transient:
        config = channel_flow_config(
            tmp_path,
            PIMPLEAlgorithm(
                type=AlgorithmType.PIMPLE,
                nCorrectors=1,
                nOuterCorrectors=3,
                residualControl={"U": 1e20},
            ),
        )
        algo = PIMPLE(create_grid(config))
    else:
        algo = SIMPLE(create_grid(simple_convergence_config(tmp_path)))

    control = algo.grid.sim_config.control.model_copy(
        update={"endTime": 3.0, "deltaT": 1.0, "writeInterval": 10}
    )
    steps = [
        runner.StepData(i, control, algo, pv.UnstructuredGrid())
        for i in range(1, 4)
    ]
    monkeypatch.setattr(runner, "manual_step", Mock(return_value=iter(steps)))
    step_spy = Mock(wraps=algo.step)
    monkeypatch.setattr(algo, "step", step_spy)
    write_spy = Mock()
    monkeypatch.setattr(runner, "save_export_fields_as_vtu", write_spy)

    runner.all_run(tmp_path / "unused.yaml")

    assert algo.has_converged()
    assert step_spy.call_count == (3 if transient else 1)
    # The final state is written even if it is off the output interval.
    write_spy.assert_called_once()


def test_simple_ignores_relative_residual_control(tmp_path: Path):
    algo = SIMPLE(create_grid(simple_convergence_config(tmp_path)))
    algo._residual_control = {  # pyright: ignore[reportPrivateUsage]
        "U": ResidualControlEntry(tolerance=1e-6, rel_tolerance=0.1)
    }
    algo._record_residual("U", 1.0)  # pyright: ignore[reportPrivateUsage]
    algo._record_residual("U", 0.01)  # pyright: ignore[reportPrivateUsage]
    assert not algo.has_converged()  # relative reduction alone is insufficient
    algo._record_residual("U", 1e-7)  # pyright: ignore[reportPrivateUsage]
    assert algo.has_simulation_converged()
