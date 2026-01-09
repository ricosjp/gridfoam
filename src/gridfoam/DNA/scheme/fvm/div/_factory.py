from __future__ import annotations

from dataclasses import dataclass

from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.div._choice import FVMDivSchemeChoice
from gridfoam.DNA.scheme.fvm.div._interface import IFVMDivOperator
from gridfoam.DNA.scheme.fvm.div._limited_linear import FVMDivLimitedLinear
from gridfoam.DNA.scheme.fvm.div._upwind import FVMDivUpwind


@dataclass(frozen=True, slots=True)
class FVMDivSchemeConfig:
    """Configuration for div scheme with field metadata."""

    choice: FVMDivSchemeChoice
    """The scheme choice."""

    def create_operator(
        self, phi_fm: FieldMeta, psi_fm: FieldMeta
    ) -> IFVMDivOperator:
        """Create an operator instance with the given field metadata."""
        match self.choice:
            case FVMDivSchemeChoice.UPWIND:
                return FVMDivUpwind(phi_fm, psi_fm)
            case FVMDivSchemeChoice.LIMITED_LINEAR:
                return FVMDivLimitedLinear(phi_fm, psi_fm)
            case _:
                raise ValueError(
                    f"Unknown div scheme choice: {self.choice.name}"
                )
