"""Physical-time replay must preserve history, gradients and caller state."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tests.helpers import channel_flow_config

from gridfoam.algorithms.checkpoint import AlgorithmCheckpoint
from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.algorithms.piso import PISO
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.factory import create_grid
from gridfoam.core.state import TensorState
from gridfoam.meta.config import (
    PIMPLEAlgorithm,
    PISOAlgorithm,
    ResidualControlEntry,
    fvSchemesConfig,
)
from gridfoam.meta.enums import AlgorithmType, DomainBoundaryPatch
from gridfoam.optimize.transient import (
    PimpleStepMap,
    PisoStepMap,
    TransientAlgorithm,
    TransientStepMap,
)


@contextmanager
def apply_inlet(
    algorithm: TransientAlgorithm, design: TensorState
) -> Generator[None]:
    boundary = DomainBoundaryPatch.X_MINUS
    original = algorithm.U.bcs[boundary]
    algorithm.U.add_boundary_conditions(
        {boundary: DirichletBC(design["inlet"])}
    )
    try:
        yield
    finally:
        algorithm.U.add_boundary_conditions({boundary: original})
        algorithm.grid.invalidate_derived_caches()


def make_map(
    tmp_path: Path, name: str = "piso", scheme: str = "backward"
) -> TransientStepMap:
    settings = (
        PISOAlgorithm(
            type=AlgorithmType.PISO, nCorrectors=2, nNonOrthogonalCorrectors=1
        )
        if name == "piso"
        else PIMPLEAlgorithm(
            type=AlgorithmType.PIMPLE,
            nCorrectors=2,
            nNonOrthogonalCorrectors=1,
            nOuterCorrectors=3,
            residualControl={},
        )
    )
    config = channel_flow_config(tmp_path, settings, refined=True)
    config = config.model_copy(
        update={
            "simulator": config.simulator.model_copy(
                update={
                    "fvSchemes": fvSchemesConfig.model_validate(
                        {"ddtSchemes": {"default": scheme}}
                    )
                }
            )
        }
    )
    grid = create_grid(config)
    if name == "piso":
        return PisoStepMap(PISO(grid), apply_inlet)
    return PimpleStepMap(PIMPLE(grid), apply_inlet)


def design(speed: torch.Tensor) -> TensorState:
    return TensorState({"inlet": speed * speed.new_tensor([1.0, 0.0, 0.0])})


def assert_ambient(
    algorithm: TransientAlgorithm, before: AlgorithmCheckpoint
) -> None:
    after = AlgorithmCheckpoint.capture(algorithm)
    assert after.iteration == before.iteration
    assert after.fields.previous_dts == before.fields.previous_dts
    assert tuple(after.fields.values) == tuple(before.fields.values)
    for key in before.fields.values:
        torch.testing.assert_close(
            after.fields.values[key], before.fields.values[key]
        )


@pytest.mark.parametrize(
    "name,scheme",
    [("piso", "euler"), ("piso", "backward"), ("pimple", "backward")],
)
def test_three_steps_match_direct_values_gradients_and_finite_differences(
    tmp_path: Path, name: str, scheme: str
) -> None:
    mapping = make_map(tmp_path, name, scheme)
    algorithm = mapping.algorithm
    baseline = AlgorithmCheckpoint.capture(algorithm)
    initial = mapping.initial_state
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    perturbation = torch.sin(algorithm.grid.cell_centers[:, :1]) * axis

    def run(parameters: torch.Tensor, *, replay: bool) -> torch.Tensor:
        velocity = initial.values["U"] + parameters[0] * perturbation
        if replay:
            blocks = dict(initial.values)
            blocks["U"] = velocity
            blocks["U/old"] = velocity
            state = initial.with_values(TensorState(blocks))
            for index in range(3):
                state = mapping.step(state, design(parameters[index + 1]))
                assert state.step_index == index + 1
                assert state.time == pytest.approx((index + 1) * state.delta_t)
                assert state.previous_dt == state.delta_t
                assert tuple(state.values) == tuple(initial.values)
            assert_ambient(algorithm, baseline)
            return torch.cat((state.values["U"].flatten(), state.values["p"]))
        baseline.restore(algorithm)
        try:
            algorithm.U.data = velocity.clone()
            algorithm.U.restore_history(velocity.clone(), None, None)
            for index in range(3):
                with apply_inlet(algorithm, design(parameters[index + 1])):
                    algorithm.step()
            return torch.cat((algorithm.U.data.flatten(), algorithm.p.data))
        finally:
            baseline.restore(algorithm)

    parameters = torch.tensor(
        [0.12, 1.1, 1.15, 1.2], dtype=torch.float64, requires_grad=True
    )
    expected = run(parameters, replay=False)
    expected_grad = torch.autograd.grad(expected.square().sum(), parameters)[0]
    actual = run(parameters, replay=True)
    actual_grad = torch.autograd.grad(actual.square().sum(), parameters)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-7, atol=1e-8)
    direction = parameters.new_tensor([0.5, 0.7, -0.3, 0.2])
    epsilon = 1e-5
    with torch.no_grad():
        plus = run(parameters + epsilon * direction, replay=True)
        minus = run(parameters - epsilon * direction, replay=True)
        fd = (plus.square().sum() - minus.square().sum()) / (2 * epsilon)
    torch.testing.assert_close(
        actual_grad @ direction, fd, rtol=3e-4, atol=1e-6
    )


def test_history_gradients_and_multiple_live_forwards(tmp_path: Path) -> None:
    mapping = make_map(tmp_path)
    speed = torch.tensor(1.2, dtype=torch.float64)
    with torch.no_grad():
        warm = mapping.step(mapping.initial_state, design(speed))
    assert warm.previous_dt is not None
    parameter = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)

    def objective(value: torch.Tensor) -> torch.Tensor:
        blocks = dict(warm.values)
        blocks["U/older"] = blocks["U/older"] + value
        blocks["phi/older"] = blocks["phi/older"] + 0.1 * value
        result = mapping.step(
            warm.with_values(TensorState(blocks)), design(speed)
        )
        return (
            result.values["U"].square().sum()
            + result.values["p"].square().sum()
        )

    first = objective(parameter)
    # Incoming ambient pollution must not affect another branch's replay.
    mapping.algorithm.HbyA.data.fill_(8.0)
    second = objective(parameter)
    assert bool((mapping.algorithm.HbyA.data == 8.0).all())
    first_grad = torch.autograd.grad(first, parameter)[0]
    second_grad = torch.autograd.grad(second, parameter)[0]
    torch.testing.assert_close(first_grad, second_grad)
    with torch.no_grad():
        fd = (objective(parameter + 1e-5) - objective(parameter - 1e-5)) / 2e-5
    torch.testing.assert_close(first_grad, fd, rtol=3e-4, atol=1e-6)


def test_fixed_counts_and_single_history_update(tmp_path: Path) -> None:
    mapping = make_map(tmp_path, "pimple")
    algorithm = mapping.algorithm
    with (
        patch.object(
            algorithm.U, "update_history", wraps=algorithm.U.update_history
        ) as u_history,
        patch.object(
            algorithm.phi, "update_history", wraps=algorithm.phi.update_history
        ) as phi_history,
        patch.object(
            algorithm.solvers["U"], "solve", wraps=algorithm.solvers["U"].solve
        ) as momentum,
        patch.object(
            algorithm.solvers["p"], "solve", wraps=algorithm.solvers["p"].solve
        ) as pressure,
    ):
        mapping.step(
            mapping.initial_state,
            design(torch.tensor(1.1, dtype=torch.float64)),
        )
    assert momentum.call_count == 3
    assert pressure.call_count == 3 * 2 * (1 + 1)
    assert u_history.call_count == phi_history.call_count == 1


@pytest.mark.parametrize("mutation", ["correctors", "outer", "residual", "dt"])
def test_changed_controls_are_rejected_before_writes(
    tmp_path: Path, mutation: str
) -> None:
    mapping = make_map(tmp_path, "pimple")
    algorithm = mapping.algorithm
    assert isinstance(algorithm, PIMPLE)
    if mutation == "correctors":
        algorithm.n_correctors += 1
    elif mutation == "outer":
        algorithm.n_outer_correctors += 1
    elif mutation == "residual":
        algorithm._residual_control["U"] = ResidualControlEntry(tolerance=1e20)  # pyright: ignore[reportPrivateUsage]
    else:
        grid = algorithm.grid
        assert isinstance(grid, AxisProjectedGrid)
        grid._sim_config = grid.sim_config.model_copy(  # pyright: ignore[reportPrivateUsage]
            update={
                "control": grid.sim_config.control.model_copy(
                    update={"deltaT": grid.dt * 2}
                )
            }
        )
    before = algorithm.U.data
    with pytest.raises(ValueError, match="controls|residualControl"):
        mapping.step(
            mapping.initial_state,
            design(torch.tensor(1.1, dtype=torch.float64)),
        )
    assert algorithm.U.data is before


def test_rejects_pimple_early_exit_at_construction(tmp_path: Path) -> None:
    config = channel_flow_config(
        tmp_path,
        PIMPLEAlgorithm(type=AlgorithmType.PIMPLE, residualControl={"U": 1e20}),
    )
    with pytest.raises(ValueError, match="residualControl"):
        PimpleStepMap(PIMPLE(create_grid(config)), apply_inlet)


def test_invalid_state_and_failed_design_restore_ambient(
    tmp_path: Path,
) -> None:
    mapping = make_map(tmp_path)
    before = AlgorithmCheckpoint.capture(mapping.algorithm)
    with pytest.raises(ValueError, match="delta_t"):
        mapping.step(
            replace(mapping.initial_state, delta_t=0.1),
            design(torch.tensor(1.1, dtype=torch.float64)),
        )
    original = mapping.algorithm.U.bcs[DomainBoundaryPatch.X_MINUS]

    @contextmanager
    def failing_design(
        algorithm: TransientAlgorithm, values: TensorState
    ) -> Generator[None]:
        with apply_inlet(algorithm, values):
            algorithm.U.data.fill_(7.0)
            raise RuntimeError("design failed")
            yield

    mapping.apply_design = failing_design
    with pytest.raises(RuntimeError, match="design failed"):
        mapping.step(
            mapping.initial_state,
            design(torch.tensor(1.1, dtype=torch.float64)),
        )
    assert_ambient(mapping.algorithm, before)
    assert mapping.algorithm.U.bcs[DomainBoundaryPatch.X_MINUS] is original
    mapping.apply_design = apply_inlet
    mapping.step(
        mapping.initial_state, design(torch.tensor(1.1, dtype=torch.float64))
    )


def test_inference_then_training(tmp_path: Path) -> None:
    mapping = make_map(tmp_path)
    with torch.inference_mode():
        state = mapping.step(
            mapping.initial_state,
            design(torch.tensor(1.2, dtype=torch.float64)),
        )
    assert not state.values["U"].requires_grad
    speed = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    mapping.step(state, design(speed)).values["U"].square().sum().backward()
    assert speed.grad is not None and torch.isfinite(speed.grad)
