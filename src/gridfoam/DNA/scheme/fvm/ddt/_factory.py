from __future__ import annotations

from dataclasses import dataclass

from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.ddt._choice import FVMDdtSchemeChoice
from gridfoam.DNA.scheme.fvm.ddt._euler import FVMDdtEuler
from gridfoam.DNA.scheme.fvm.ddt._interface import IFVMDdtOperator


@dataclass(frozen=True, slots=True)
class FVMDdtSchemeConfig:
    """Configuration for ddt scheme with field metadata."""

    choice: FVMDdtSchemeChoice
    """The scheme choice."""

    def create_operator(self, psi_fm: FieldMeta) -> IFVMDdtOperator:
        """Create an operator instance with the given field metadata."""
        match self.choice:
            case FVMDdtSchemeChoice.EULER:
                return FVMDdtEuler(psi_fm)
            case _:
                raise ValueError(
                    f"Unknown ddt scheme choice: {self.choice.name}"
                )

