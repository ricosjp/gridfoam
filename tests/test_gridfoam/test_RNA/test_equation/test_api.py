"""Tests for equation API."""

import pytest

from gridfoam.DNA.ASTNodes.operator_node import OperatorNode
from gridfoam.DNA.enum import OperatorType
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.RNA.equation.api import ddt, div, grad, laplacian


def test_ddt():
    """Test ddt function."""
    field = FieldMeta(name="U", label="Velocity")
    result = ddt(field)
    assert isinstance(result, OperatorNode)
    assert result.type == OperatorType.DDT
    assert result.args == [field]


def test_div_single_arg():
    """Test div function with single argument."""
    field = FieldMeta(name="U", label="Velocity")
    result = div(field)
    assert isinstance(result, OperatorNode)
    assert result.type == OperatorType.DIV
    assert result.args == [field]


def test_div_two_args():
    """Test div function with two arguments."""
    field1 = FieldMeta(name="U", label="Velocity")
    field2 = FieldMeta(name="T", label="Temperature")
    result = div(field1, field2)
    assert isinstance(result, OperatorNode)
    assert result.type == OperatorType.DIV
    assert result.args == [field1, field2]


def test_laplacian():
    """Test laplacian function."""
    gamma = FieldMeta(name="gamma", label="Gamma")
    psi = FieldMeta(name="psi", label="Psi")
    result = laplacian(gamma, psi)
    assert isinstance(result, OperatorNode)
    assert result.type == OperatorType.LAPLACIAN
    assert result.args == [gamma, psi]


def test_grad():
    """Test grad function."""
    field = FieldMeta(name="p", label="Pressure")
    result = grad(field)
    assert isinstance(result, OperatorNode)
    assert result.type == OperatorType.GRAD
    assert result.args == [field]
