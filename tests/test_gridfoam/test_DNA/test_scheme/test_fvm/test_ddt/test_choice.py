from __future__ import annotations

from gridfoam.DNA.scheme.fvm.ddt._choice import (
    DdtSchemeChoice,
    match_ddt_choice,
)
from gridfoam.DNA.scheme.fvm.ddt._euler import FVMEulerDdt


class TestMatchDdtChoice:
    """Test suite for match_ddt_choice function."""

    def test_match_euler(self) -> None:
        """Test that match_ddt_choice returns FVMEulerDdt for EULER."""
        result = match_ddt_choice(DdtSchemeChoice.EULER)

        assert isinstance(result, FVMEulerDdt)
