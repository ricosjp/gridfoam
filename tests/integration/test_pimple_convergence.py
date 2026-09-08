"""PIMPLE convergence must use fresh residuals within each time step."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from tests.helpers import channel_flow_config

from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.core.grid.factory import create_grid
from gridfoam.meta.config import PIMPLEAlgorithm, ResidualControlEntry
from gridfoam.meta.enums import AlgorithmType


@pytest.mark.parametrize("relative", [False, True])
def test_velocity_convergence_uses_current_outer_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool
) -> None:
    """
    PIMPLE exits on current velocity residuals and refreshes the relative
    baseline each step.
    """
    config = PIMPLEAlgorithm(
        type=AlgorithmType.PIMPLE,
        nCorrectors=1,
        nOuterCorrectors=5,
        residualControl={
            "U": ResidualControlEntry(
                # Tight absTol so the relative case can only exit via relTol.
                tolerance=1e-8 if relative else 0.1,
                rel_tolerance=0.1 if relative else 0.0,
            )
        },
    )
    algo = PIMPLE(create_grid(channel_flow_config(tmp_path, config)))
    # Control only the residual measurement; run the real momentum and
    # pressure solves so the test exercises step() and its early exit.
    residual = Mock(side_effect=[1.0, 0.05, 0.04, 0.1, 0.02, 0.005, 0.004])
    monkeypatch.setattr(
        "gridfoam.algorithms.pimple.field_initial_residual", residual
    )

    algo.step()
    assert algo.has_converged()
    # Includes the final outer iteration after convergence is detected.
    assert residual.call_count == 3

    if relative:
        # The new baseline is 0.1, so 0.02 must not pass a 10% reduction
        # (absTol is too tight to exit on its own). Reusing the previous
        # baseline of 1.0 would accept 0.02 and exit too early.
        algo.step()
        assert algo.has_converged()
        assert residual.call_count == 7


def test_pimple_clears_convergence_after_an_unconverged_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later unconverged time step clears the previous convergence flag."""
    config = PIMPLEAlgorithm(
        type=AlgorithmType.PIMPLE,
        nCorrectors=1,
        nOuterCorrectors=3,
        residualControl={"U": 0.1},
    )
    algo = PIMPLE(create_grid(channel_flow_config(tmp_path, config)))
    residual = Mock(side_effect=[0.01, 0.005, 1.0, 0.8, 0.5])
    monkeypatch.setattr(
        "gridfoam.algorithms.pimple.field_initial_residual", residual
    )

    algo.step()
    assert algo.has_converged()
    assert residual.call_count == 2
    algo.step()
    assert not algo.has_converged()
    assert residual.call_count == 5


@pytest.mark.parametrize("n_outer", [1, 2, 3, 4])
def test_relative_check_skips_initial_and_scheduled_final_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_outer: int
) -> None:
    """
    Relative control cannot trigger on the initial or scheduled final outer
    iteration.
    """
    config = PIMPLEAlgorithm(
        type=AlgorithmType.PIMPLE,
        nCorrectors=1,
        nOuterCorrectors=n_outer,
        residualControl={
            "U": ResidualControlEntry(tolerance=1e-8, rel_tolerance=2.0)
        },
    )
    algo = PIMPLE(create_grid(channel_flow_config(tmp_path, config)))
    residual = Mock(return_value=1.0)
    monkeypatch.setattr(
        "gridfoam.algorithms.pimple.field_initial_residual", residual
    )
    algo.step()
    # Even relTol > 1 cannot trigger on the first residual. Checking at
    # the scheduled final iteration is also disabled in v2606.
    assert algo.has_converged() == (n_outer == 4)
    assert residual.call_count == min(n_outer, 3)


def test_pressure_uses_first_baseline_and_last_solve_initial_residual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Pressure convergence retains the first baseline and latest pre-solve
    residual.
    """
    config = PIMPLEAlgorithm(
        type=AlgorithmType.PIMPLE,
        nCorrectors=2,
        nNonOrthogonalCorrectors=1,
        nOuterCorrectors=4,
        residualControl={"p": 0.1},
    )
    algo = PIMPLE(create_grid(channel_flow_config(tmp_path, config)))
    residual = Mock(side_effect=[1.0, 0.3, 0.2, 0.01, 0.1, 0.05, 0.02, 0.005])
    monkeypatch.setattr(
        "gridfoam.algorithms.utils.pressure_correction.field_initial_residual",
        residual,
    )
    algo.step()
    assert algo.has_converged()
    assert residual.call_count == 8  # one outer pass plus final pass
    assert algo._initial_residuals["p"] == 1.0  # pyright: ignore[reportPrivateUsage]
    assert algo._current_residuals["p"] == 0.005  # pyright: ignore[reportPrivateUsage]
