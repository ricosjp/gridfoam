"""SIMPLE step maps must be replayable and differentiable in explicit inputs."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from tests.helpers import channel_flow_config

from gridfoam.algorithms.checkpoint import AlgorithmCheckpoint
from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.grid.factory import create_grid
from gridfoam.core.state import TensorState
from gridfoam.meta.config import RelaxationFactorsConfig, SIMPLEAlgorithm
from gridfoam.meta.enums import AlgorithmType, DomainBoundaryPatch
from gridfoam.optimize.simple import SimpleStepMap


@contextmanager
def _apply_inlet_velocity(
    algorithm: SIMPLE, design: TensorState
) -> Generator[None]:
    """Apply and restore the test's uniform inlet velocity."""
    patch = DomainBoundaryPatch.X_MINUS
    value = design["inlet_velocity"]
    if value.shape != (3,):
        raise ValueError("inlet_velocity must have shape (3,)")
    if (
        value.dtype != algorithm.grid.dtype
        or value.device != algorithm.grid.device
    ):
        raise ValueError("inlet_velocity must match grid dtype and device")
    original = algorithm.U.bcs.get(patch)
    if not isinstance(original, DirichletBC):
        raise ValueError("The inlet patch must have a Dirichlet velocity BC")
    algorithm.U.add_boundary_conditions({patch: DirichletBC(value)})
    try:
        yield
    finally:
        algorithm.U.add_boundary_conditions({patch: original})
        algorithm.grid.invalidate_derived_caches()


def make_step_map(tmp_path: Path) -> SimpleStepMap:
    """Create a fresh SIMPLE step map for an isolated channel-flow case."""
    settings = SIMPLEAlgorithm(
        type=AlgorithmType.SIMPLE,
        relaxationFactors=RelaxationFactorsConfig(
            equations={"U": 0.7, "p": 0.3}
        ),
    )
    return SimpleStepMap(
        SIMPLE(
            create_grid(channel_flow_config(tmp_path, settings, refined=False))
        ),
        _apply_inlet_velocity,
    )


def test_step_replays_auxiliary_fields_and_restores_ambient_state(
    tmp_path: Path,
) -> None:
    """A step replays complete state and restores all ambient mutations."""
    mapping = make_step_map(tmp_path)
    algo = mapping.algorithm
    design = TensorState(
        {"inlet_velocity": torch.tensor([1.2, 0.0, 0.0], dtype=algo.grid.dtype)}
    )
    before = AlgorithmCheckpoint.capture(algo)
    original_bc = algo.U.bcs[DomainBoundaryPatch.X_MINUS]
    with torch.no_grad(), mapping.apply_design(algo, design):
        algo.step()
    expected = AlgorithmCheckpoint.capture(algo)
    before.restore(algo)
    with torch.no_grad():
        actual = mapping.step(mapping.initial_state, design)
    for name, field in (
        ("U", algo.U),
        ("p", algo.p),
        ("HbyA", algo.HbyA),
        ("rAtU", algo.rAtU),
    ):
        torch.testing.assert_close(
            actual[name], expected.fields.values[f"cell/{field.name}/data"]
        )
    after = AlgorithmCheckpoint.capture(algo)
    assert after.iteration == before.iteration
    for key in before.fields.values:
        torch.testing.assert_close(
            after.fields.values[key], before.fields.values[key]
        )
    assert algo.U.bcs[DomainBoundaryPatch.X_MINUS] is original_bc
    # Incoming scratch pollution must not become a hidden argument of G.
    algo.HbyA.data.fill_(3.0)
    algo.rAtU.data.fill_(4.0)
    again = mapping.step(mapping.initial_state, design)
    for key in actual:
        torch.testing.assert_close(again[key], actual[key])
    assert torch.all(algo.HbyA.data == 3.0)


def test_step_design_gradient_matches_finite_difference_on_repeated_replays(
    tmp_path: Path,
) -> None:
    """Repeated replay gradients agree with centered finite differences."""
    mapping = make_step_map(tmp_path)
    dtype = mapping.algorithm.grid.dtype
    direction = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    for speed in (1.1, 1.2):
        parameter = torch.tensor(speed, dtype=dtype, requires_grad=True)

        def objective(value: torch.Tensor) -> torch.Tensor:
            result = mapping.step(
                mapping.initial_state,
                TensorState({"inlet_velocity": value * direction}),
            )
            return (
                result["U"].square().sum() + 0.01 * result["p"].square().sum()
            )

        loss = objective(parameter)
        gradient = torch.autograd.grad(loss, parameter)[0]
        with torch.no_grad():
            fd = (
                objective(parameter + 1e-5) - objective(parameter - 1e-5)
            ) / 2e-5
        torch.testing.assert_close(gradient, fd, rtol=2e-4, atol=1e-6)


def test_auxiliary_state_gradient_matches_finite_difference(
    tmp_path: Path,
) -> None:
    """Auxiliary-state gradients agree with centered finite differences."""
    mapping = make_step_map(tmp_path)
    design = TensorState(
        {"inlet_velocity": torch.tensor([1.1, 0.0, 0.0], dtype=torch.float64)}
    )
    with torch.no_grad():
        state = mapping.step(mapping.initial_state, design)
    perturbation = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)

    def objective(value: torch.Tensor) -> torch.Tensor:
        blocks = dict(state)
        blocks["phiHbyA"] = state["phiHbyA"] + value
        result = mapping.step(TensorState(blocks), design)
        return result["p"].square().sum() + result["U"].square().sum()

    grad = torch.autograd.grad(objective(perturbation), perturbation)[0]
    with torch.no_grad():
        fd = (
            objective(perturbation + 1e-6) - objective(perturbation - 1e-6)
        ) / 2e-6
    torch.testing.assert_close(grad, fd, rtol=2e-4, atol=1e-6)


def test_design_failure_restores_fields_and_boundary(tmp_path: Path) -> None:
    """An invalid design leaves fields and boundary conditions unchanged."""
    mapping = make_step_map(tmp_path)
    before = AlgorithmCheckpoint.capture(mapping.algorithm)
    with pytest.raises(ValueError, match="shape"):
        mapping.step(
            mapping.initial_state,
            TensorState({"inlet_velocity": torch.ones(2)}),
        )
    after = AlgorithmCheckpoint.capture(mapping.algorithm)
    for key in before.fields.values:
        torch.testing.assert_close(
            after.fields.values[key], before.fields.values[key]
        )
