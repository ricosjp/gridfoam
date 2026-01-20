"""Tests for FVM laplacian scheme choice."""


from gridfoam.DNA.scheme.fvm.laplacian._choice import FVMLaplacianSchemeChoice


def test_fvm_laplacian_scheme_choice_enum():
    """Test FVMLaplacianSchemeChoice enum values."""
    assert FVMLaplacianSchemeChoice.LINEAR.value == "GaussLinear"
