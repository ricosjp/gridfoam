from dataclasses import dataclass

from gridfoam.DNA.ASTNodes._interface import IASTNode
from gridfoam.DNA.enum import FieldRole
from gridfoam.DNA.meta.boundary_condition import BoundaryConditionMeta
from gridfoam.DNA.meta.field import FieldMeta


@dataclass(slots=True)
class EquationMeta:
    # Identification
    name: str  # e.g. "momentum equation", "energy equation"

    # Reference to the target field (meta)
    target_field: FieldMeta
    boundary_conditions: list[BoundaryConditionMeta]

    # Abstract Syntax Tree (AST) root node
    ast_root: IASTNode

    def __post_init__(self) -> None:
        if self.target_field.role != FieldRole.STATE:
            raise ValueError(
                f"Target field {self.target_field.name} must be a state field"
            )
