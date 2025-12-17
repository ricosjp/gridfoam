from __future__ import annotations

from dataclasses import dataclass

from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.grad._choice import FVMGradSchemeChoice
from gridfoam.DNA.scheme.fvm.grad._interface import IFVMGradOperator
from gridfoam.DNA.scheme.fvm.grad._linear import FVMGradLinear


@dataclass(frozen=True, slots=True)
class FVMGradSchemeConfig:
    """Configuration for grad scheme with field metadata."""

    choice: FVMGradSchemeChoice
    """The scheme choice."""

    def create_operator(self, psi_fm: FieldMeta) -> IFVMGradOperator:
        """Create an operator instance with the given field metadata."""
        match self.choice:
            case FVMGradSchemeChoice.LINEAR:
                return FVMGradLinear(psi_fm)
            case _:
                raise ValueError(
                    f"Unknown grad scheme choice: {self.choice.name}"
                )
