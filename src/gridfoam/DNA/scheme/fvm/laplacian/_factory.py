from __future__ import annotations

from dataclasses import dataclass

from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.laplacian._choice import FVMLaplacianSchemeChoice
from gridfoam.DNA.scheme.fvm.laplacian._interface import IFVMLaplacianOperator
from gridfoam.DNA.scheme.fvm.laplacian._linear import FVMLaplacianLinear


@dataclass(frozen=True, slots=True)
class FVMLaplacianSchemeConfig:
    """Configuration for laplacian scheme with field metadata."""

    choice: FVMLaplacianSchemeChoice
    """The scheme choice."""

    def create_operator(
        self, gamma_fm: FieldMeta, psi_fm: FieldMeta
    ) -> IFVMLaplacianOperator:
        """Create an operator instance with the given field metadata."""
        match self.choice:
            case FVMLaplacianSchemeChoice.LINEAR:
                return FVMLaplacianLinear(gamma_fm, psi_fm)
            case _:
                raise ValueError(
                    f"Unknown laplacian scheme choice: {self.choice.name}"
                )
