"""Tests for FieldMeta."""

import pytest
import torch

from gridfoam.DNA.ASTNodes.arithmetic_node import ArithmeticNode, ArithmeticType
from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.field import FieldMeta


def test_field_meta_basic():
    """Test basic FieldMeta creation."""
    field = FieldMeta(
        name="test",
        label="Test Field",
        description="Test description",
    )
    assert field.name == "test"
    assert field.label == "Test Field"
    assert field.description == "Test description"
    assert field.role == FieldRole.STATE
    assert field.layout == FieldLayout.CELL
    assert field.components == 1
    assert field.time_levels == 2  # STATE fields have 2 time levels


def test_field_meta_auxiliary_role():
    """Test FieldMeta with AUXILIARY role."""
    field = FieldMeta(
        name="test",
        label="Test Field",
        role=FieldRole.AUXILIARY,
    )
    assert field.time_levels == 1  # Non-STATE fields have 1 time level


def test_field_meta_initialize_func_cell():
    """Test FieldMeta with initialize_func for cell-centered field."""

    def init_func(x, y, z):
        return torch.zeros(1, *x.shape)

    field = FieldMeta(
        name="test",
        label="Test Field",
        layout=FieldLayout.CELL,
        initialize_func=init_func,
    )
    assert field.initialize_func == init_func


def test_field_meta_initialize_func_face_error():
    """Test FieldMeta raises error for initialize_func on face-centered field."""

    def init_func(x, y, z):
        return torch.zeros(1, *x.shape)

    with pytest.raises(ValueError, match="initialize_func is only supported"):
        FieldMeta(
            name="test",
            label="Test Field",
            layout=FieldLayout.FACE,
            initialize_func=init_func,
        )


def test_field_meta_add():
    """Test FieldMeta addition."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    result = field1 + field2
    assert isinstance(result, ArithmeticNode)
    assert result.type == ArithmeticType.ADD
    assert result.arg1 == field1
    assert result.arg2 == field2


def test_field_meta_add_none():
    """Test FieldMeta addition with None."""
    field = FieldMeta(name="field", label="Field")
    result = field + None
    assert result == field


def test_field_meta_sub():
    """Test FieldMeta subtraction."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    result = field1 - field2
    assert isinstance(result, ArithmeticNode)
    assert result.type == ArithmeticType.SUB


def test_field_meta_key():
    """Test FieldMeta key property."""
    field = FieldMeta(name="test_field", label="Test Field")
    assert field.key == "test_field"
