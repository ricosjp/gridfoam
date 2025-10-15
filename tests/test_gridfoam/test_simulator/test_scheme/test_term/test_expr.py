import pathlib
from unittest.mock import Mock

import pytest
import torch
from jaxtyping import Float

from gridfoam import TensorGrid
from gridfoam._interface._fvmterm import IFVMTerm
from gridfoam._simulator._scheme._term._expr import Expr
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


class MockFVMTerm(IFVMTerm):
    """Mock FVM term for testing."""

    def __init__(
        self,
        name: str,
        matvec_result: torch.Tensor = None,
        diag_result: torch.Tensor = None,
        rhs_result: torch.Tensor = None,
    ):
        self.name = name
        self.matvec_result = (
            matvec_result
            if matvec_result is not None
            else torch.tensor([1.0, 2.0, 3.0])
        )
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
        return self.matvec_result

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


class TestExpr:
    """Test Expr class."""

    def test_expr_init_addition(self) -> None:
        """Test Expr initialization with addition operator."""
        left = MockFVMTerm("left")
        right = MockFVMTerm("right")
        expr = Expr(left, right, "+")

        assert expr.left == left
        assert expr.right == right
        assert expr.op == "+"

    def test_expr_init_subtraction(self) -> None:
        """Test Expr initialization with subtraction operator."""
        left = MockFVMTerm("left")
        right = MockFVMTerm("right")
        expr = Expr(left, right, "-")

        assert expr.left == left
        assert expr.right == right
        assert expr.op == "-"

    def test_expr_matvec_addition(self) -> None:
        """Test Expr.matvec with addition operator."""
        # Create mock terms with known results
        left_result = torch.tensor([1.0, 2.0, 3.0])
        right_result = torch.tensor([4.0, 5.0, 6.0])

        left = MockFVMTerm("left", matvec_result=left_result)
        right = MockFVMTerm("right", matvec_result=right_result)
        expr = Expr(left, right, "+")

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        x = torch.tensor([1.0, 2.0, 3.0])
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test matvec
        result = expr.matvec(octree_level, x, dt, dx)
        expected = left_result + right_result
        torch.testing.assert_close(result, expected)

    def test_expr_matvec_subtraction(self) -> None:
        """Test Expr.matvec with subtraction operator."""
        # Create mock terms with known results
        left_result = torch.tensor([5.0, 6.0, 7.0])
        right_result = torch.tensor([1.0, 2.0, 3.0])

        left = MockFVMTerm("left", matvec_result=left_result)
        right = MockFVMTerm("right", matvec_result=right_result)
        expr = Expr(left, right, "-")

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        x = torch.tensor([1.0, 2.0, 3.0])
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test matvec
        result = expr.matvec(octree_level, x, dt, dx)
        expected = left_result - right_result
        torch.testing.assert_close(result, expected)

    def test_expr_matvec_invalid_operator(self) -> None:
        """Test Expr.matvec with invalid operator."""
        left = MockFVMTerm("left")
        right = MockFVMTerm("right")
        expr = Expr(left, right, "*")  # Invalid operator

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        x = torch.tensor([1.0, 2.0, 3.0])
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test that invalid operator raises ValueError
        with pytest.raises(ValueError, match="Unknown operator"):
            expr.matvec(octree_level, x, dt, dx)

    def test_expr_diag_addition(self) -> None:
        """Test Expr.diag method."""
        # Create mock terms with known diagonal results
        left_diag = torch.tensor([2.0, 3.0, 4.0])
        right_diag = torch.tensor([1.0, 1.0, 1.0])

        left = MockFVMTerm("left", diag_result=left_diag)
        right = MockFVMTerm("right", diag_result=right_diag)
        expr = Expr(left, right, "+")

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test diag (always adds regardless of operator)
        result = expr.diag(octree_level, dt, dx)
        expected = left_diag + right_diag
        torch.testing.assert_close(result, expected)

    def test_expr_diag_subtraction(self) -> None:
        """Test Expr.diag method."""
        # Create mock terms with known diagonal results
        left_diag = torch.tensor([5.0, 6.0, 7.0])
        right_diag = torch.tensor([1.0, 2.0, 3.0])

        left = MockFVMTerm("left", diag_result=left_diag)
        right = MockFVMTerm("right", diag_result=right_diag)
        expr = Expr(left, right, "-")

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test diag (always adds regardless of operator)
        result = expr.diag(octree_level, dt, dx)
        expected = left_diag - right_diag
        torch.testing.assert_close(result, expected)

    def test_expr_diag_invalid_operator(self) -> None:
        """Test Expr.diag with invalid operator."""
        left = MockFVMTerm("left")
        right = MockFVMTerm("right")
        expr = Expr(left, right, "*")  # Invalid operator

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test that invalid operator raises ValueError
        with pytest.raises(ValueError, match="Unknown operator"):
            expr.diag(octree_level, dt, dx)

    def test_expr_rhs_addition(self) -> None:
        """Test Expr.rhs with addition operator."""
        # Create mock terms with known RHS results
        left_rhs = torch.tensor([1.0, 2.0, 3.0])
        right_rhs = torch.tensor([0.5, 1.0, 1.5])

        left = MockFVMTerm("left", rhs_result=left_rhs)
        right = MockFVMTerm("right", rhs_result=right_rhs)
        expr = Expr(left, right, "+")

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test rhs
        result = expr.rhs(octree_level, dt, dx)
        expected = left_rhs + right_rhs
        torch.testing.assert_close(result, expected)

    def test_expr_rhs_subtraction(self) -> None:
        """Test Expr.rhs with subtraction operator."""
        # Create mock terms with known RHS results
        left_rhs = torch.tensor([3.0, 4.0, 5.0])
        right_rhs = torch.tensor([1.0, 1.0, 1.0])

        left = MockFVMTerm("left", rhs_result=left_rhs)
        right = MockFVMTerm("right", rhs_result=right_rhs)
        expr = Expr(left, right, "-")

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test rhs
        result = expr.rhs(octree_level, dt, dx)
        expected = left_rhs - right_rhs
        torch.testing.assert_close(result, expected)

    def test_expr_rhs_invalid_operator(self) -> None:
        """Test Expr.rhs with invalid operator."""
        left = MockFVMTerm("left")
        right = MockFVMTerm("right")
        expr = Expr(left, right, "/")  # Invalid operator

        # Mock octree level and parameters
        octree_level = Mock(spec=PyOctreeLevel)
        dt = 0.1
        dx = torch.tensor([0.1, 0.1, 0.1])

        # Test that invalid operator raises ValueError
        with pytest.raises(ValueError, match="Unknown operator"):
            expr.rhs(octree_level, dt, dx)
