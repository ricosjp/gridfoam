"""Tests for OperatorNode."""

from gridfoam.DNA.ASTNodes.arithmetic_node import ArithmeticNode, ArithmeticType
from gridfoam.DNA.ASTNodes.operator_node import OperatorNode
from gridfoam.DNA.enum import OperatorType
from gridfoam.DNA.meta.field import FieldMeta


def test_operator_node_basic():
    """Test basic OperatorNode creation."""
    field = FieldMeta(name="U", label="Velocity")
    node = OperatorNode(
        type=OperatorType.DIV,
        args=[field],
    )
    assert node.type == OperatorType.DIV
    assert node.args == [field]
    assert node.operator is None


def test_operator_node_add():
    """Test OperatorNode addition."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    node = OperatorNode(
        type=OperatorType.GRAD,
        args=[field1],
    )
    result = node + field2
    assert isinstance(result, ArithmeticNode)
    assert result.type == ArithmeticType.ADD


def test_operator_node_sub():
    """Test OperatorNode subtraction."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    node = OperatorNode(
        type=OperatorType.LAPLACIAN,
        args=[field1],
    )
    result = node - field2
    assert isinstance(result, ArithmeticNode)
    assert result.type == ArithmeticType.SUB


def test_operator_node_key():
    """Test OperatorNode key property."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    node = OperatorNode(
        type=OperatorType.DIV,
        args=[field1, field2],
    )
    assert node.key == "div(field1, field2)"
