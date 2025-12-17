from __future__ import annotations

from gridfoam.DNA.scheme.fvm.div._choice import (
    DivSchemeChoice,
    match_div_choice,
)
from gridfoam.DNA.scheme.fvm.div._upwind import FVMUpwindDiv


class TestMatchDivChoice:
    """Test suite for match_div_choice function."""

    def test_match_upwind(self) -> None:
        """Test that match_div_choice returns FVMUpwindDiv for UPWIND."""
        result = match_div_choice(DivSchemeChoice.UPWIND)

        assert isinstance(result, FVMUpwindDiv)
