from dataclasses import dataclass
from enum import Enum

from gridfoam.DNA.ASTNodes._interface import IASTNode


class ArithmeticType(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"

@dataclass(slots=True)
class ArithmeticNode(IASTNode):
    type: ArithmeticType
    arg1: IASTNode
    arg2: IASTNode

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
        return f"{self.arg1.key} {self.type.value} {self.arg2.key}"

