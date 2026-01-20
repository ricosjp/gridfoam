"""Tests for FVMatrix."""

import torch

from gridfoam.DNA.enum import Axis
from gridfoam.DNA.fielddata._cell import CellField
from gridfoam.DNA.fielddata._fvmatrix import FVMatrix


def test_fvmatrix_init():
    """Test FVMatrix initialization."""
    matrix = FVMatrix(
        C=3, N=8, H=2, dtype=torch.float32, device=torch.device("cpu")
    )
    assert matrix.C == 3
    assert matrix.N == 8
    assert matrix.H == 2
    assert matrix.dtype == torch.float32
    assert matrix.device == torch.device("cpu")
    assert isinstance(matrix.a_P, CellField)
    assert isinstance(matrix.source, CellField)


def test_fvmatrix_setters():
    """Test FVMatrix property setters."""
    matrix = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    value = torch.ones(1, 4, 4, 4) * 2.0
    matrix.a_P = value
    matrix.a_E = value
    matrix.source = value
    assert torch.allclose(matrix.a_P.interior[0], value)
    assert torch.allclose(matrix.a_E.interior[0], value)
    assert torch.allclose(matrix.source.interior[0], value)


def test_fvmatrix_add():
    """Test FVMatrix addition."""
    matrix1 = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    matrix2 = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    value1 = torch.ones(1, 4, 4, 4) * 2.0
    value2 = torch.ones(1, 4, 4, 4) * 3.0
    matrix1.a_P = value1
    matrix2.a_P = value2
    result = matrix1 + matrix2
    assert torch.allclose(result.a_P.interior[0], torch.ones(1, 4, 4, 4) * 5.0)


def test_fvmatrix_sub():
    """Test FVMatrix subtraction."""
    matrix1 = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    matrix2 = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    value1 = torch.ones(1, 4, 4, 4) * 5.0
    value2 = torch.ones(1, 4, 4, 4) * 3.0
    matrix1.a_P = value1
    matrix2.a_P = value2
    result = matrix1 - matrix2
    assert torch.allclose(result.a_P.interior[0], torch.ones(1, 4, 4, 4) * 2.0)


def test_fvmatrix_iadd():
    """Test FVMatrix in-place addition."""
    matrix1 = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    matrix2 = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    value1 = torch.ones(1, 4, 4, 4) * 2.0
    value2 = torch.ones(1, 4, 4, 4) * 3.0
    matrix1.a_P = value1
    matrix2.a_P = value2
    matrix1 += matrix2
    assert torch.allclose(matrix1.a_P.interior[0], torch.ones(1, 4, 4, 4) * 5.0)


def test_fvmatrix_apply():
    """Test FVMatrix apply method."""
    matrix = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    matrix.a_P = torch.ones(1, 4, 4, 4) * 2.0
    xi = CellField(
        T=1, C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    xi.interior[0] = torch.ones(1, 4, 4, 4) * 3.0
    result = matrix.apply(xi)
    assert result.shape == (1, 4, 4, 4)
    # For uniform field and coefficients, result should be 2.0 * 3.0 = 6.0
    # (plus contributions from neighbors which are zero in this case)
    assert torch.allclose(result, torch.ones(1, 4, 4, 4) * 6.0)


def test_fvmatrix_get_coeff_along():
    """Test FVMatrix get_coeff_along method."""
    matrix = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    coeff = matrix.get_coeff_along(Axis.X, forward=True)
    assert coeff == matrix.a_E
    coeff = matrix.get_coeff_along(Axis.X, forward=False)
    assert coeff == matrix.a_W


def test_fvmatrix_set_coeff_along():
    """Test FVMatrix set_coeff_along method."""
    matrix = FVMatrix(
        C=1, N=4, H=1, dtype=torch.float32, device=torch.device("cpu")
    )
    value = torch.ones(1, 4, 4, 4) * 5.0
    matrix.set_coeff_along(Axis.X, forward=True, value=value)
    assert torch.allclose(matrix.a_E.interior[0], value)
    matrix.set_coeff_along(Axis.X, forward=False, value=value)
    assert torch.allclose(matrix.a_W.interior[0], value)
