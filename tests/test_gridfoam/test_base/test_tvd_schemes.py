import pytest
import torch

from gridfoam._base._tvd_scheme import tvd_scheme
from gridfoam._base._tvd_schemes import (
    BoundedLinear,
    Minmod,
    Superbee,
    Upwind,
    VanAlbada,
    VanLeer,
)
from gridfoam.utils.enums import TVDScheme


class TestTVDSchemes:
    """Test TVD scheme implementations."""

    def test_bounded_linear_correction_term(self) -> None:
        """Test BoundedLinear correction term."""
        scheme = BoundedLinear()

        # Test with positive gradients
        delta_minus = torch.tensor([1.0, 2.0, 3.0])
        delta_plus = torch.tensor([2.0, 1.0, 4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction >= 0)

        # Test with negative gradients
        delta_minus = torch.tensor([-1.0, -2.0, -3.0])
        delta_plus = torch.tensor([-2.0, -1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction <= 0)

        # Test with mixed signs (should be zero)
        delta_minus = torch.tensor([1.0, -2.0, 3.0])
        delta_plus = torch.tensor([-2.0, 1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Should be zero when signs are different
        assert torch.allclose(correction, torch.zeros_like(correction))

    def test_minmod_correction_term(self) -> None:
        """Test Minmod correction term."""
        scheme = Minmod()

        # Test with positive gradients
        delta_minus = torch.tensor([1.0, 2.0, 3.0])
        delta_plus = torch.tensor([2.0, 1.0, 4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction >= 0)

        # Test with negative gradients
        delta_minus = torch.tensor([-1.0, -2.0, -3.0])
        delta_plus = torch.tensor([-2.0, -1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction <= 0)

        # Test with mixed signs (should be zero)
        delta_minus = torch.tensor([1.0, -2.0, 3.0])
        delta_plus = torch.tensor([-2.0, 1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Should be zero when signs are different
        assert torch.allclose(correction, torch.zeros_like(correction))

    def test_superbee_correction_term(self) -> None:
        """Test Superbee correction term."""
        scheme = Superbee()

        # Test with positive gradients
        delta_minus = torch.tensor([1.0, 2.0, 3.0])
        delta_plus = torch.tensor([2.0, 1.0, 4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction >= 0)

        # Test with negative gradients
        delta_minus = torch.tensor([-1.0, -2.0, -3.0])
        delta_plus = torch.tensor([-2.0, -1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction <= 0)

        # Test with mixed signs (should be zero)
        delta_minus = torch.tensor([1.0, -2.0, 3.0])
        delta_plus = torch.tensor([-2.0, 1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Should be zero when signs are different
        assert torch.allclose(correction, torch.zeros_like(correction))

    def test_upwind_correction_term(self) -> None:
        """Test Upwind correction term."""
        scheme = Upwind()

        # Test with any input - should always return zero
        delta_minus = torch.tensor([1.0, -2.0, 3.0])
        delta_plus = torch.tensor([2.0, 1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Should always be zero
        assert torch.allclose(correction, torch.zeros_like(correction))

    def test_van_albada_correction_term(self) -> None:
        """Test VanAlbada correction term."""
        scheme = VanAlbada()

        # Test with positive gradients
        delta_minus = torch.tensor([1.0, 2.0, 3.0])
        delta_plus = torch.tensor([2.0, 1.0, 4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction >= 0)

        # Test with negative gradients
        delta_minus = torch.tensor([-1.0, -2.0, -3.0])
        delta_plus = torch.tensor([-2.0, -1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction <= 0)

        # Test with mixed signs (should be zero)
        delta_minus = torch.tensor([1.0, -2.0, 3.0])
        delta_plus = torch.tensor([-2.0, 1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Should be zero when signs are different
        assert torch.allclose(correction, torch.zeros_like(correction))

    def test_van_leer_correction_term(self) -> None:
        """Test VanLeer correction term."""
        scheme = VanLeer()

        # Test with positive gradients
        delta_minus = torch.tensor([1.0, 2.0, 3.0])
        delta_plus = torch.tensor([2.0, 1.0, 4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction >= 0)

        # Test with negative gradients
        delta_minus = torch.tensor([-1.0, -2.0, -3.0])
        delta_plus = torch.tensor([-2.0, -1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Check that correction is non-negative
        assert torch.all(correction <= 0)

        # Test with mixed signs (should be zero)
        delta_minus = torch.tensor([1.0, -2.0, 3.0])
        delta_plus = torch.tensor([-2.0, 1.0, -4.0])
        correction = scheme.correction_term(delta_minus, delta_plus)

        # Should be zero when signs are different
        assert torch.allclose(correction, torch.zeros_like(correction))

    def test_tvd_scheme_factory(self) -> None:
        """Test TVD scheme factory function."""
        # Test all valid schemes
        schemes = [
            TVDScheme.SUPERBEE,
            TVDScheme.MINMOD,
            TVDScheme.BOUNDED_LINEAR,
            TVDScheme.VAN_LEER,
            TVDScheme.VAN_ALBADA,
            TVDScheme.UPWIND,
        ]

        expected_classes = [
            Superbee,
            Minmod,
            BoundedLinear,
            VanLeer,
            VanAlbada,
            Upwind,
        ]

        for scheme, expected_class in zip(
            schemes, expected_classes, strict=False
        ):
            instance = tvd_scheme(scheme)
            assert isinstance(instance, expected_class)

        # Test invalid scheme
        with pytest.raises(ValueError, match="Invalid scheme"):
            tvd_scheme("INVALID_SCHEME")  # type: ignore

    def test_correction_term_edge_cases(self) -> None:
        """Test correction terms with edge cases."""
        schemes = [
            BoundedLinear(),
            Minmod(),
            Superbee(),
            VanAlbada(),
            VanLeer(),
        ]

        # Test with zero gradients
        delta_minus = torch.tensor([0.0, 0.0, 0.0])
        delta_plus = torch.tensor([0.0, 0.0, 0.0])

        for scheme in schemes:
            correction = scheme.correction_term(delta_minus, delta_plus)
            assert torch.allclose(correction, torch.zeros_like(correction))

        # Test with very small gradients
        delta_minus = torch.tensor([1e-10, 1e-10, 1e-10])
        delta_plus = torch.tensor([1e-10, 1e-10, 1e-10])

        for scheme in schemes:
            correction = scheme.correction_term(delta_minus, delta_plus)
            assert torch.all(correction >= 0)

        # Test with very large gradients
        delta_minus = torch.tensor([1e10, 1e10, 1e10])
        delta_plus = torch.tensor([1e10, 1e10, 1e10])

        for scheme in schemes:
            correction = scheme.correction_term(delta_minus, delta_plus)
            assert torch.all(correction >= 0)
