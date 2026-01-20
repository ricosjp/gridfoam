"""Tests for ArithmeticNode."""


from gridfoam.DNA.ASTNodes.arithmetic_node import ArithmeticNode, ArithmeticType
from gridfoam.DNA.meta.field import FieldMeta


def test_arithmetic_node_add():
    """Test ArithmeticNode addition."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    node = ArithmeticNode(
        type=ArithmeticType.ADD,
        arg1=field1,
        arg2=field2,
    )
    result = node + field1
    assert isinstance(result, ArithmeticNode)
    assert result.type == ArithmeticType.ADD


def test_arithmetic_node_add_none():
    """Test ArithmeticNode addition with None."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    node = ArithmeticNode(
        type=ArithmeticType.ADD,
        arg1=field1,
        arg2=field2,
    )
    result = node + None
    assert result == node


def test_arithmetic_node_sub():
    """Test ArithmeticNode subtraction."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    node = ArithmeticNode(
        type=ArithmeticType.SUB,
        arg1=field1,
        arg2=field2,
    )
    result = node - field1
    assert isinstance(result, ArithmeticNode)
    assert result.type == ArithmeticType.SUB


def test_arithmetic_node_key():
    """Test ArithmeticNode key property."""
    field1 = FieldMeta(name="field1", label="Field 1")
    field2 = FieldMeta(name="field2", label="Field 2")
    node = ArithmeticNode(
        type=ArithmeticType.ADD,
        arg1=field1,
        arg2=field2,
    )
    assert node.key == "field1 + field2"


def test_arithmetic_type_enum():
    """Test ArithmeticType enum."""
    assert ArithmeticType.ADD.value == "+"
    assert ArithmeticType.SUB.value == "-"
    assert ArithmeticType.MUL.value == "*"
