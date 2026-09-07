"""Physical tensor axes survive FV operators, history, and linear solves."""

import pytest
import torch
from tests.helpers import refined_3d_grid

from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.equation import equation
from gridfoam.core.field import CellField, FaceField, get_or_create_cellfield
from gridfoam.core.fvmatrix import FvMatrix
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.fv import fvc, fvm
from gridfoam.meta.config import SolverConfig
from gridfoam.meta.enums import (
    DomainBoundaryPatch,
    FieldRole,
    GradScheme,
    SolverType,
)
from gridfoam.solvers.factory import create_solver


@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
@pytest.mark.parametrize("scheme", [GradScheme.LINEAR, GradScheme.LEASTSQUARE])
def test_linear_tensor_field_operators(
    shape: tuple[int, ...], scheme: GradScheme
):
    grid = refined_3d_grid(scheme)
    q = CellField(grid, "tensor_q", FieldRole.TRANSIENT, shape)
    generator = torch.Generator().manual_seed(123)
    jacobian = torch.randn((*shape, 3), generator=generator, dtype=grid.dtype)
    offset = torch.randn(shape, generator=generator, dtype=grid.dtype)

    def exact(points: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nj,...j->n...", points, jacobian) + offset

    q.data = exact(grid.cell_centers)
    q.add_boundary_conditions(
        {
            patch: NeumannBC(
                (2 * (patch.to_direction().value % 2) - 1)
                * jacobian[..., patch.to_direction().value // 2]
            )
            for patch in DomainBoundaryPatch
        }
    )
    grad = fvc.grad(q)
    assert grad.component_shape == (*shape, 3)
    torch.testing.assert_close(
        grad.data, jacobian.expand_as(grad.data), atol=1e-11, rtol=1e-11
    )
    face = fvc.interpolate(q)
    torch.testing.assert_close(
        face.single_data,
        exact(grid.face_centers[face.single_mask]),
        atol=1e-11,
        rtol=1e-11,
    )
    torch.testing.assert_close(
        face.domain_bnd_data,
        exact(grid.domain_bnd_face_centers),
        atol=1e-11,
        rtol=1e-11,
    )
    sn = fvc.sn_grad(q)
    expected_sn = jacobian.movedim(-1, 0)[grid.axis[sn.single_mask]]
    torch.testing.assert_close(
        sn.single_data, expected_sn, atol=1e-11, rtol=1e-11
    )
    diffusion = fvm.laplacian(0.2, q)
    torch.testing.assert_close(
        diffusion.flux(q.data),
        0.2
        * torch.einsum(
            "n,n...->n...",
            torch.linalg.vector_norm(grid.Sf[sn.single_mask], dim=-1),
            sn.single_data,
        ),
        atol=1e-11,
        rtol=1e-11,
    )
    # A linear field has zero Laplacian including the prescribed boundaries.
    torch.testing.assert_close(
        diffusion.multiply(q.data), diffusion.source, atol=1e-11, rtol=1e-11
    )
    q.update_history(reset=True)
    q.data = q.data + 1
    transient = fvm.ddt(q)
    expected_ddt = (
        (grid.cell_volumes / grid.dt)
        .reshape((-1,) + (1,) * len(shape))
        .expand_as(q.data)
    )
    torch.testing.assert_close(
        transient.multiply(q.data) - transient.source, expected_ddt
    )
    packed = face.pack().clone()
    face.unpack(packed)
    assert packed.shape == (face.packed_n_rows(), *shape)
    torch.testing.assert_close(face.pack(), packed)
    q.sync_to_grid_topology(topology_changed=True)
    face.sync_to_grid_topology(topology_changed=True)
    assert q.data.shape == q.old_data.shape == (grid.num_cells, *shape)
    assert face.pack().shape == packed.shape


@pytest.mark.parametrize("shape", [(), (3,), (3, 3)])
@pytest.mark.parametrize(
    "method", [SolverType.CG, SolverType.BiCGSTAB, SolverType.PyAMG]
)
def test_tensor_solve_and_adjoint_match_dense(
    shape: tuple[int, ...],
    method: SolverType,
    small_axis_projected_grid: AxisProjectedGrid,
):
    grid = small_axis_projected_grid
    q = CellField(grid, "tensor_solve", FieldRole.LOCAL, shape)
    mat = FvMatrix(q)
    mat.diag = torch.full_like(mat.diag, 10.0, requires_grad=True)
    mat.upper = torch.full_like(mat.upper, -0.2, requires_grad=True)
    mat.lower = torch.full_like(mat.lower, -0.2, requires_grad=True)
    rhs = (
        torch.linspace(0.1, 1.3, q.data.numel(), dtype=grid.dtype)
        .reshape(q.data.shape)
        .requires_grad_()
    )
    mat.source = rhs
    solver = create_solver(
        SolverConfig(
            method=method, tolerance=1e-12, rel_tolerance=0.0, max_iter=1000
        )
    )
    result = solver.solve(equation(q, mat))
    assert result.solution.shape == q.data.shape
    assert len(result.stats) == q.num_components
    assert all(stat.converged for stat in result.stats)
    dense = torch.diag(mat.diag)
    dense = dense.index_put(
        (grid.owner, grid.neighbour), mat.upper, accumulate=True
    )
    dense = dense.index_put(
        (grid.neighbour, grid.owner), mat.lower, accumulate=True
    )
    expected = torch.linalg.solve(
        dense, rhs.reshape(grid.num_cells, -1)
    ).reshape(rhs.shape)
    torch.testing.assert_close(
        result.solution, expected, atol=1e-10, rtol=1e-10
    )
    inputs = (mat.diag, mat.upper, mat.lower, rhs)
    actual_grads = torch.autograd.grad(
        result.solution.square().sum(), inputs, retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)


def test_reject_legacy_shapes(small_axis_projected_grid: AxisProjectedGrid):
    grid = small_axis_projected_grid
    scalar = CellField(grid, "shape_scalar", FieldRole.LOCAL, ())
    with pytest.raises(ValueError, match="shape"):
        scalar.data = torch.zeros((grid.num_cells, 1), dtype=grid.dtype)
    with pytest.raises(ValueError, match="shape"):
        scalar.reset_data([0.0])
    with pytest.raises(ValueError, match="component_shape"):
        CellField(grid, "invalid_shape", FieldRole.LOCAL, (9,))
    with pytest.raises(ValueError, match="component_shape"):
        get_or_create_cellfield(grid, scalar.name, FieldRole.LOCAL, (3,))
    tensor = FaceField(grid, "shape_tensor", FieldRole.LOCAL, (3, 3))
    with pytest.raises(ValueError, match="shape"):
        tensor.unpack(
            torch.zeros((tensor.packed_n_rows(), 9), dtype=grid.dtype)
        )
