"""Tests for TVD scheme interface."""

import pytest

from gridfoam.DNA.scheme.tvd._interface import ITVDScheme


def test_itvd_scheme_is_abstract():
    """Test that ITVDScheme is abstract."""
    with pytest.raises(TypeError):
        ITVDScheme()


def test_itvd_scheme_has_correction_term():
    """Test that ITVDScheme has correction_term method."""
    assert hasattr(ITVDScheme, "correction_term")
    assert ITVDScheme.correction_term.__isabstractmethod__
