"""Tests for enum module."""

import pytest

from gridfoam.DNA.enum import (
    Axis,
    BoundaryConditionType,
    FieldLayout,
    FieldRole,
    NormType,
    OperatorType,
    Precision,
    TimeLevel,
)


def test_field_role():
    """Test FieldRole enum."""
    assert FieldRole.STATE != FieldRole.AUXILIARY
    assert FieldRole.AUXILIARY != FieldRole.DIAGNOSTIC
    assert FieldRole.DIAGNOSTIC != FieldRole.STATE


def test_field_layout():
    """Test FieldLayout enum."""
    assert FieldLayout.CELL != FieldLayout.FACE


def test_precision():
    """Test Precision enum."""
    assert Precision.FLOAT32 != Precision.FLOAT64


def test_axis():
    """Test Axis enum."""
    assert Axis.X != Axis.Y
    assert Axis.Y != Axis.Z
    assert Axis.Z != Axis.X


def test_norm_type():
    """Test NormType enum."""
    assert NormType.L_inf.value == "L_inf"
    assert NormType.L_2.value == "L_2"


def test_time_level():
    """Test TimeLevel enum."""
    assert TimeLevel.CUR != TimeLevel.OLD
    assert TimeLevel.OLD != TimeLevel.OLD_OLD
    assert TimeLevel.OLD_OLD != TimeLevel.CUR


def test_boundary_condition_type():
    """Test BoundaryConditionType enum."""
    assert BoundaryConditionType.DIRICHLET != BoundaryConditionType.NEUMANN


def test_operator_type():
    """Test OperatorType enum."""
    assert OperatorType.DDT.value == "ddt"
    assert OperatorType.DIV.value == "div"
    assert OperatorType.GRAD.value == "grad"
    assert OperatorType.LAPLACIAN.value == "laplacian"
