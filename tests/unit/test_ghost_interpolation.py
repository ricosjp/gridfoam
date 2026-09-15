"""What the GCIBM interpolation prototype guarantees.

Transpose
    ``apply`` and ``transpose_apply`` are adjoints, including repeated donors
    and trailing physical axes. Missing donors are rejected; an empty image
    set still has a well-defined transpose.

Manufactured Poisson
    Ghost constraints of the form ``u_g + interpolate(u) = 2 u_wall`` give
    the linear solution, and the wall-position adjoint matches autograd.
    The system is non-symmetric, so the untransposed solve is the wrong
    adjoint.
"""

import pytest
import torch

from gridfoam.fv.kernels.ghost_interpolation import InterpolationStencil


def test_transpose_matches_autograd_with_repeated_donors() -> None:
    """Repeated indices and a vector field share the same transpose."""
    gen = torch.Generator().manual_seed(42)
    indices = torch.tensor([[0, 2, 2], [2, 1, 0]])
    weights = torch.tensor(
        [[0.2, 0.3, 0.5], [0.8, 0.4, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    op = InterpolationStencil(indices, weights, n_cells=4)
    x = torch.randn(
        (4, 3), generator=gen, dtype=torch.float64, requires_grad=True
    )
    y = torch.randn((2, 3), generator=gen, dtype=torch.float64)
    lhs = (op.apply(x) * y).sum()
    rhs = (x * op.transpose_apply(y)).sum()
    torch.testing.assert_close(lhs, rhs)
    dx, _dw = torch.autograd.grad(lhs, (x, weights))
    torch.testing.assert_close(dx, op.transpose_apply(y))


def _poisson_matrix(walls: torch.Tensor) -> torch.Tensor:
    # Ghost centers: 0 and 1. Fluid centers: 0.25, 0.5 and 0.75.
    # The images reflected across the walls lie between fluid donors.
    images = torch.stack((2 * walls[0], 2 * walls[1] - 1))
    left = (images[0] - 0.25) / 0.25
    right = (images[1] - 0.5) / 0.25
    weights = torch.stack(
        (torch.stack((1 - left, left)), torch.stack((1 - right, right)))
    )
    op = InterpolationStencil(torch.tensor([[1, 2], [2, 3]]), weights, 5)
    eye = torch.eye(5, dtype=walls.dtype)
    ghost_rows = eye[[0, 4]] + op.apply(eye)
    laplacian = (-eye[:-2] + 2 * eye[1:-1] - eye[2:]) / 0.25**2
    return torch.cat((ghost_rows[:1], laplacian, ghost_rows[1:]), dim=0)


def test_ghost_poisson_solution_and_wall_position_adjoint() -> None:
    """Linear solution is exact; wall gradients need the transposed system."""
    walls = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor([2.0, 0.0, 0.0, 0.0, 6.0], dtype=torch.float64)
    matrix = _poisson_matrix(walls)
    solution = torch.linalg.solve(matrix, rhs)
    x = torch.linspace(0, 1, 5, dtype=walls.dtype)
    exact = 1 + 2 * (x - walls[0]) / (walls[1] - walls[0])
    torch.testing.assert_close(solution, exact)
    objective_weights = torch.tensor(
        [0.0, 0.2, 0.3, 0.5, 0.0], dtype=walls.dtype
    )
    grad = torch.autograd.grad(
        solution @ objective_weights, walls, retain_graph=True
    )[0]
    adjoint = torch.linalg.solve(matrix.detach().T, objective_weights)
    implicit = torch.autograd.grad(
        matrix @ solution.detach(), walls, grad_outputs=-adjoint
    )[0]
    torch.testing.assert_close(grad, implicit)
    assert not torch.allclose(matrix, matrix.T)


def test_negative_donor_indices_are_rejected() -> None:
    """``-1`` must not wrap to the last cell."""
    with pytest.raises(ValueError, match="out of bounds"):
        InterpolationStencil(torch.tensor([[0, -1]]), torch.ones(1, 2), 3)


def test_empty_image_set_has_a_zero_transpose() -> None:
    """No images still maps cotangents back onto the cell axis."""
    op = InterpolationStencil(
        torch.empty((0, 2), dtype=torch.long), torch.empty(0, 2), 3
    )
    assert op.apply(torch.ones(3, 3)).shape == (0, 3)
    torch.testing.assert_close(
        op.transpose_apply(torch.empty(0, 3)), torch.zeros(3, 3)
    )
