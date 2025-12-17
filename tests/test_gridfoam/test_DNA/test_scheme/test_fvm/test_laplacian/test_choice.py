from __future__ import annotations

from gridfoam.DNA.scheme.fvm.laplacian._choice import (
    LaplacianSchemeChoice,
    match_laplacian_choice,
)
from gridfoam.DNA.scheme.fvm.laplacian._linear import FVMLinearLaplacian


class TestMatchDivChoice:
    """Test suite for match_div_choice function."""

    def test_match_upwind(self) -> None:
        """Test that match_div_choice returns FVMUpwindDiv for UPWIND."""
        result = match_laplacian_choice(LaplacianSchemeChoice.LINEAR)

        assert isinstance(result, FVMLinearLaplacian)
