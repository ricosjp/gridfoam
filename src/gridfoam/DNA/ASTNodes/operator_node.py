from dataclasses import dataclass

from gridfoam.DNA.ASTNodes._interface import IASTNode
from gridfoam.DNA.ASTNodes.arithmetic_node import ArithmeticNode, ArithmeticType
from gridfoam.DNA.enum import OperatorType
from gridfoam.DNA.scheme.fvm._interface import IFVMOperator


@dataclass(slots=True)
class OperatorNode(IASTNode):
    type: OperatorType
    args: list[IASTNode]
    operator: IFVMOperator | None = None

    def __add__(self, other: IASTNode | None) -> IASTNode:
        if other is None:
            return self
        return ArithmeticNode(type=ArithmeticType.ADD, arg1=self, arg2=other)

    def __sub__(self, other: IASTNode | None) -> IASTNode:
        if other is None:
            return self
        return ArithmeticNode(type=ArithmeticType.SUB, arg1=self, arg2=other)

    @property
    def key(self) -> str:
        return f"{self.type.value}({', '.join([arg.key for arg in self.args])})"
