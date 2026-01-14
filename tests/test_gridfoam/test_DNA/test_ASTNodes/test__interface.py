"""Tests for AST node interface."""

import pytest

from gridfoam.DNA.ASTNodes._interface import IASTNode


def test_iast_node_is_abstract():
    """Test that IASTNode is abstract."""
    with pytest.raises(TypeError):
        IASTNode()


def test_iast_node_has_add_method():
    """Test that IASTNode has __add__ method."""
    assert hasattr(IASTNode, "__add__")
    assert IASTNode.__add__.__isabstractmethod__


def test_iast_node_has_sub_method():
    """Test that IASTNode has __sub__ method."""
    assert hasattr(IASTNode, "__sub__")
    assert IASTNode.__sub__.__isabstractmethod__


def test_iast_node_has_key_property():
    """Test that IASTNode has key property."""
    assert hasattr(IASTNode, "key")
