"""What checkpoints guarantee for in-memory replay.

Replay
    SIMPLE, PISO and PIMPLE can restore a captured step, including an extra
    diffusion scalar. PISO + BDF2 covers two-level velocity and flux history.

Iteration snapshots
    Captured residual dictionaries are copies. Later ``step`` calls must not
    mutate the checkpoint. PISO has empty residuals; PIMPLE has outer flags.

Layout and graphs
    ``with_values`` rejects a different layout. Restore keeps autograd links
    and does not write into the saved tensors.

Boundaries and history
    Design inputs such as Dirichlet values are reapplied after restore.
    ``replace_packed`` and history restore do not mutate autograd leaves.

Stale checkpoints
    Equal-size IBM updates and remeshes, new registered fields, and nested
    config edits fail before any field buffer is written.
"""

from pathlib import Path

import pytest
import torch
from tests.conftest import small_gridfoam_config
from tests.helpers import channel_flow_config

from gridfoam.algorithms.checkpoint import AlgorithmCheckpoint
from gridfoam.algorithms.factory import create_algorithm
from gridfoam.algorithms.iteration_state import IterationState
from gridfoam.algorithms.pimple import PIMPLE
from gridfoam.algorithms.piso import PISO
from gridfoam.algorithms.simple import SIMPLE
from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.core.checkpoint import GridCheckpoint
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import GridBase
from gridfoam.core.grid.factory import create_grid
from gridfoam.core.state import TensorState
from gridfoam.fv import fvm
from gridfoam.fv.boundary_ops import iter_boundary_states
from gridfoam.fv.flux import correct_flux
from gridfoam.meta.config import (
    PIMPLEAlgorithm,
    PISOAlgorithm,
    RelaxationFactorsConfig,
    SIMPLEAlgorithm,
    SolverConfig,
    fvSchemesConfig,
)
from gridfoam.meta.enums import (
    AlgorithmType,
    DomainBoundaryPatch,
    FieldRole,
    MeshMotion,
    SolverType,
)
from gridfoam.solvers.factory import create_solver

_Settings = SIMPLEAlgorithm | PISOAlgorithm | PIMPLEAlgorithm


def _algorithm_settings(name: str) -> _Settings:
    if name == "simple":
        return SIMPLEAlgorithm(
            type=AlgorithmType.SIMPLE,
            residualControl={"U": 10.0},
            relaxationFactors=RelaxationFactorsConfig(
                equations={"U": 0.7, "p": 0.3}
            ),
        )
    if name == "piso":
        return PISOAlgorithm(type=AlgorithmType.PISO, nCorrectors=2)
    return PIMPLEAlgorithm(
        type=AlgorithmType.PIMPLE,
        nCorrectors=2,
        nOuterCorrectors=3,
        residualControl={"U": 10.0},
    )


def _make_algorithm(grid: GridBase, name: str) -> SIMPLE | PISO | PIMPLE:
    if name == "simple":
        return SIMPLE(grid)
    if name == "piso":
        return PISO(grid)
    return PIMPLE(grid)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("piso", IterationState(7)),
        (
            "pimple",
            IterationState(7, {"U": 1.0}, {"U": 0.1}, True),
        ),
    ],
    ids=["piso-empty", "pimple-residuals"],
)
def test_captured_iteration_state_is_isolated_from_later_steps(
    tmp_path: Path,
    name: str,
    expected: IterationState,
) -> None:
    """Checkpoint residuals stay frozen after ``step`` writes new values."""
    algo = create_algorithm(
        create_grid(channel_flow_config(tmp_path, _algorithm_settings(name)))
    )
    algo.restore_iteration_state(expected)
    saved = AlgorithmCheckpoint.capture(algo)
    assert saved.iteration == expected
    algo.restore_iteration_state(IterationState(99))
    with torch.no_grad():
        algo.step()
    assert saved.iteration == expected
    saved.restore(algo)
    assert algo.capture_iteration_state() == expected


def test_with_values_rejects_layout_changes_and_keeps_the_graph() -> None:
    """Replacement tensors must match the checkpoint; restore clones them."""
    grid = create_grid(small_gridfoam_config())
    field = CellField(grid, "T", FieldRole.TRANSIENT, ())
    saved = GridCheckpoint.capture(grid)
    key = "cell/T/data"
    invalid = dict(saved.values)
    invalid[key] = torch.zeros(1, dtype=grid.dtype, device=grid.device)
    with pytest.raises(ValueError, match="layout"):
        saved.with_values(TensorState(invalid))
    with pytest.raises(ValueError, match="keys or order"):
        saved.with_values(
            TensorState(dict(reversed(tuple(saved.values.items()))))
        )

    parameter = torch.tensor(2.0, dtype=grid.dtype, requires_grad=True)
    updated = dict(saved.values)
    updated[key] = torch.ones_like(field.data) * parameter
    saved.with_values(TensorState(updated)).restore(grid)
    field.data.sum().backward()
    torch.testing.assert_close(
        parameter.grad, parameter.new_tensor(grid.num_cells)
    )
    torch.testing.assert_close(saved.values[key], torch.zeros_like(field.data))


@pytest.mark.parametrize(
    "name,scheme",
    [
        ("simple", "euler"),
        ("piso", "euler"),
        ("pimple", "euler"),
        ("piso", "backward"),
    ],
)
def test_restored_algorithm_step_matches_the_original_step(
    tmp_path: Path, name: str, scheme: str
) -> None:
    """One captured step plus an extra scalar reproduces fields and history."""
    config = channel_flow_config(tmp_path, _algorithm_settings(name))
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
    algo = _make_algorithm(grid, name)
    # Start away from the uniform channel fixed point so replay exercises
    # evolving velocity, pressure and flux as well as scalar transport.
    axis = torch.tensor([1.0, 0.0, 0.0], dtype=grid.dtype)
    algo.U.data = algo.U.data + 0.1 * torch.sin(grid.cell_centers[:, :1]) * axis
    algo.p.data = 0.025 * torch.sin(grid.cell_centers[:, 0])
    correct_flux(algo.phi, algo.U, update_internal=True)
    algo.U.update_history(reset=True)
    algo.phi.update_history(reset=True)
    temperature = CellField(grid, "T", FieldRole.TRANSIENT, ())
    temperature.data = torch.sin(grid.cell_centers[:, 0])
    temperature.update_history(reset=True)
    scalar_solver = create_solver(
        SolverConfig(method=SolverType.CG, tolerance=1e-12, rel_tolerance=0.0)
    )

    def step() -> None:
        algo.step()
        matrix = fvm.ddt(temperature) - fvm.laplacian(0.1, temperature)
        temperature.data = scalar_solver.solve(
            equation(temperature, matrix)
        ).solution
        temperature.update_history()

    with torch.no_grad():
        step()
        step()  # Establish two real time levels before taking a checkpoint.
    initial = AlgorithmCheckpoint.capture(algo)
    assert "cell/T/older" in initial.fields.values
    if scheme == "backward":
        assert "face/phi/older" in initial.fields.values
        assert "cell/U/older" in initial.fields.values
    with torch.no_grad():
        step()
    expected = AlgorithmCheckpoint.capture(algo)
    temperature.data.fill_(9.0)
    with torch.no_grad():
        step()
    initial.restore(algo)
    assert algo.capture_iteration_state() == initial.iteration
    assert grid.fv_cache.face_geometry is None
    with torch.no_grad():
        step()
    actual = AlgorithmCheckpoint.capture(algo)
    assert actual.iteration == expected.iteration
    assert actual.fields.previous_dts == expected.fields.previous_dts
    for key, value in expected.fields.values.items():
        torch.testing.assert_close(
            actual.fields.values[key], value, rtol=1e-10, atol=1e-12
        )


def test_restore_rebuilds_boundary_graphs_for_a_new_design_value() -> None:
    """Dirichlet values are inputs: restore must not reuse a cached graph."""
    grid = create_grid(small_gridfoam_config())
    field = CellField(grid, "T", FieldRole.TRANSIENT, ())
    bc = DirichletBC(torch.tensor(0.0, dtype=grid.dtype))
    field.add_boundary_conditions({DomainBoundaryPatch.X_MINUS: bc})
    saved = GridCheckpoint.capture(grid)
    with torch.no_grad():
        list(iter_boundary_states(field))
    for value in (2.0, 3.0):
        inlet = torch.tensor(value, dtype=grid.dtype, requires_grad=True)
        saved.restore(grid)
        bc.value = inlet
        loss = torch.stack(
            [
                state.psi_b.square().sum()
                for state in iter_boundary_states(field)
            ]
        ).sum()
        loss.backward()
        n_faces = int(
            grid.get_domain_bnd_mask(DomainBoundaryPatch.X_MINUS).sum()
        )
        torch.testing.assert_close(
            inlet.grad, torch.tensor(2 * value * n_faces, dtype=grid.dtype)
        )


def test_restore_keeps_history_and_packed_face_autograd_leaves() -> None:
    """History restore and ``replace_packed`` must not write into leaves."""
    grid = create_grid(small_gridfoam_config())
    field = CellField(grid, "T", FieldRole.TRANSIENT, ())
    flux = FaceField(grid, "test_phi", FieldRole.LOCAL, ())
    previous = torch.randn_like(field.data, requires_grad=True)
    older = torch.randn_like(field.data, requires_grad=True)
    field.restore_history(previous, older, 0.02)
    field.data = torch.ones_like(field.data, requires_grad=True)
    packed = flux.pack().detach().requires_grad_()
    flux.replace_packed(packed)
    checkpoint = GridCheckpoint.capture(grid, detach=False)
    checkpoint.restore(grid)
    assert field.older_data is not None
    loss = field.old_data.sum() + 2 * field.older_data.sum() + flux.pack().sum()
    loss.backward()
    torch.testing.assert_close(previous.grad, torch.ones_like(previous))
    torch.testing.assert_close(older.grad, 2 * torch.ones_like(older))
    torch.testing.assert_close(packed.grad, torch.ones_like(packed))
    assert field.previous_dt == 0.02


@pytest.mark.parametrize(
    "remesh,topology",
    [(False, 0), (True, 1)],
    ids=["ibm-update", "remesh"],
)
def test_equal_size_geometry_updates_reject_old_checkpoints(
    remesh: bool,
    topology: int,
) -> None:
    """Same cell count is not enough after IBM update or remesh."""
    config = small_gridfoam_config()
    config = config.model_copy(
        update={
            "fluxel": config.fluxel.model_copy(
                update={"motion": MeshMotion.DYNAMIC}
            )
        }
    )
    grid = create_grid(config)
    assert isinstance(grid, AxisProjectedGrid)
    field = CellField(grid, "T", FieldRole.TRANSIENT, ())
    saved = GridCheckpoint.capture(grid)
    if remesh:
        grid.remesh(warn_outside_refinement=False)
    else:
        grid.update_ib(
            translation=[0.01, 0.0, 0.0], warn_outside_refinement=False
        )
    assert field.data.shape == saved.values["cell/T/data"].shape
    assert grid.geometry_revision == 1
    assert grid.topology_revision == topology
    with pytest.raises(ValueError, match="geometry has changed"):
        saved.restore(grid)


def test_registry_changes_fail_before_restoring_any_field() -> None:
    """A new registered field must not leave previously written buffers."""
    grid = create_grid(small_gridfoam_config())
    first = CellField(grid, "T", FieldRole.TRANSIENT, ())
    saved = GridCheckpoint.capture(grid)
    second = CellField(grid, "k", FieldRole.LOCAL, ())
    first.data.fill_(7.0)
    with pytest.raises(ValueError, match="fields have changed"):
        saved.restore(grid)
    assert torch.all(first.data == 7)
    assert grid.get_cellfield("k") is second


def test_nested_configuration_changes_invalidate_checkpoint() -> None:
    """Mutable nested solver dicts count as a configuration change."""
    grid = create_grid(small_gridfoam_config())
    field = CellField(grid, "T", FieldRole.TRANSIENT, ())
    saved = GridCheckpoint.capture(grid)
    field.data.fill_(7.0)
    grid.sim_config.fvSolution.solvers["p"] = SolverConfig(
        method=SolverType.CG, tolerance=1e-7
    )
    with pytest.raises(ValueError, match="configuration or time step"):
        saved.restore(grid)
    assert torch.all(field.data == 7.0)
