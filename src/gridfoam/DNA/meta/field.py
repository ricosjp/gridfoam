from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Float

from gridfoam.DNA.ASTNodes._interface import IASTNode
from gridfoam.DNA.ASTNodes.arithmetic_node import ArithmeticNode, ArithmeticType
from gridfoam.DNA.enum import FieldLayout, FieldRole


@dataclass(slots=True)
class FieldMeta(IASTNode):
    # Identification
    name: str  # e.g. "U", "p", "T"
    label: str  # e.g. "Velocity", "Pressure"
    description: str = ""

    # Physical meaning
    role: FieldRole = FieldRole.STATE
    layout: FieldLayout = FieldLayout.CELL
    components: int = (
        1  # How many components the field has (e.g. 3 for vector fields)
    )
    unit: str = ""  # e.g. "m/s", "Pa", "K"

    # Numerical attributes
    requires_grad: bool = False  # Whether to require gradient for the field
    dtype: torch.dtype = torch.float64
    time_levels: int = 1  # How many time levels to hold (new/old)

    # Initializer (only for cell-centered fields)
    initialize_func: (
        Callable[
            [
                Float[torch.Tensor, "..."],  # x (W W W)
                Float[torch.Tensor, "..."],  # y (W W W)
                Float[torch.Tensor, "..."],  # z (W W W)
            ],
            Float[torch.Tensor, "..."],  # (C W W W)
        ]
        | None
    ) = None

    # Configuration generation hints
    default_output: bool = True  # Whether to output by default in VTK etc.

    def __post_init__(self) -> None:
        if self.role is FieldRole.STATE:
            self.time_levels = 2
        else:
            self.time_levels = 1

        if (
            self.initialize_func is not None
            and self.layout is not FieldLayout.CELL
        ):
            raise ValueError(
                "initialize_func is only supported for cell-centered fields"
            )

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
        return self.name
