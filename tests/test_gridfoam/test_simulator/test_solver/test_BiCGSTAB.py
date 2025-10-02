import pathlib
from unittest.mock import Mock

import pytest
import torch
from jaxtyping import Float

from gridfoam import TensorGrid
from gridfoam._interface._fvmterm import FVMTerm
from gridfoam._simulator._scheme._term._expr import Expr
from gridfoam._simulator._solver.BiCGSTAB import BiCGSTAB
from gridfoam.cubion import PyOctreeLevel


@pytest.fixture
def config_path():
    """Create a config."""
    config_path = pathlib.Path("tests/data/yaml/Debug.yaml")
    return config_path


@pytest.fixture
def grid(config_path: pathlib.Path):
    """Create a test grid with cell fields."""
    grid = TensorGrid.build(config_path)
    grid.add_cell_field("U", (3,), torch.float32)
    grid.add_cell_field("T", (1,), torch.float32)
    grid.allocate_field_tensors()
    return grid


class MockFVMTerm(FVMTerm):
    """Mock FVM term for testing."""

    def __init__(
        self,
        name: str,
        diag_result: torch.Tensor = None,
        rhs_result: torch.Tensor = None,
    ):
        self.name = name
        self.diag_result = (
            diag_result
            if diag_result is not None
            else torch.tensor([1.0, 1.0, 1.0])
        )
        self.rhs_result = (
            rhs_result
            if rhs_result is not None
            else torch.tensor([0.5, 1.0, 1.5])
        )

    def matvec(
        self,
        octree_level: PyOctreeLevel,
        x: torch.Tensor,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        return x

    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        return self.diag_result

    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        return self.rhs_result


class TestBiCGSTAB:
    """Test BiCGSTAB class."""

    def test_bicgstab_init(self) -> None:
        """Test BiCGSTAB initialization."""
        # Create mock expression
        left = MockFVMTerm("left")
        right = MockFVMTerm("right")
        expr = Expr(left, right, "+")

        # Initialize BiCGSTAB solver
        bicgstab = BiCGSTAB(expr)

        assert bicgstab.expr == expr
        assert bicgstab.max_iter == 1000
        assert bicgstab.tol == 1e-6

    def test_bicgstab_configure(self) -> None:
        """Test BiCGSTAB configuration."""
        # Create mock expression
        left = MockFVMTerm("left")
        right = MockFVMTerm("right")
        expr = Expr(left, right, "+")

        # Initialize BiCGSTAB solver
        bicgstab = BiCGSTAB(expr)

        # Configure solver
        bicgstab.configure(max_iter=500, tol=1e-8)

        assert bicgstab.max_iter == 500
        assert bicgstab.tol == 1e-8

    def test_bicgstab_solve_simple_convergence(self) -> None:
        """Test BiCGSTAB solve with simple convergence case."""
        # Create a simple identity-like system: A*x = b where A ≈ I
        n = 3
        identity_diag = torch.tensor([1.0, 1.0, 1.0])
        rhs = torch.tensor([0.5, 1.0, 1.5])

        # Create mock expression that behaves like identity
        expr = MockFVMTerm(
            "identity", identity_diag, rhs
        )

        # Initialize BiCGSTAB solver
        bicgstab = BiCGSTAB(expr)
        bicgstab.configure(max_iter=10, tol=1e-6)

        # Mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.n_leaf_cells = n

        # Solve
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # This should converge quickly since it's essentially solving x = 0
        result = bicgstab.solve(octree_level, dt, dx)

        assert result.shape == (n,)
        assert torch.allclose(result, rhs, atol=1e-5)

    def test_bicgstab_solve_zero_diagonal_error(self) -> None:
        """Test BiCGSTAB solve with zero diagonal elements."""
        # Create expression with zero diagonal
        n = 3
        zero_diag = torch.tensor([0.0, 1.0, 1.0])
        expr = MockFVMTerm("zero_diag", diag_result=zero_diag)

        # Initialize BiCGSTAB solver
        bicgstab = BiCGSTAB(expr)

        # Mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.n_leaf_cells = n

        # Solve should raise ValueError
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        with pytest.raises(ValueError, match="diag is 0"):
            bicgstab.solve(octree_level, dt, dx)

    def test_bicgstab_solve_no_convergence(self) -> None:
        """Test BiCGSTAB solve with no convergence."""
        # Create a system that won't converge easily
        n = 3
        # Use a matrix that's not well-conditioned
        bad_diag = torch.tensor([0.01, 2.0, 0.2])
        bad_rhs = torch.tensor([15.0, 1.0, 6.0])

        expr = MockFVMTerm("bad_conditioned", bad_diag, bad_rhs)

        # Initialize BiCGSTAB solver with very strict tolerance
        bicgstab = BiCGSTAB(expr)
        bicgstab.configure(max_iter=5, tol=1e-12)  # Very strict

        # Mock octree level
        octree_level = Mock(spec=PyOctreeLevel)
        octree_level.n_leaf_cells = n

        # Solve should raise ValueError for no convergence
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        with pytest.raises(ValueError, match="BiCGSTAB did not converge"):
            bicgstab.solve(octree_level, dt, dx)
