"""Tests for TVD scheme choice and factory."""

from unittest.mock import Mock

import pytest
import torch

from gridfoam.DNA.scheme.tvd._choice import TVDFactory, TVDSchemeChoice
from gridfoam.DNA.scheme.tvd._interface import ITVDScheme
from gridfoam.DNA.scheme.tvd._limited_linear import LimitedLinear
from gridfoam.DNA.scheme.tvd._minmod import Minmod
from gridfoam.DNA.scheme.tvd._superbee import Superbee
from gridfoam.DNA.scheme.tvd._upwind import Upwind
from gridfoam.DNA.scheme.tvd._van_albada import VanAlbada
from gridfoam.DNA.scheme.tvd._van_leer import VanLeer


def test_tvd_scheme_choice_enum():
    """Test TVDSchemeChoice enum values."""
    assert TVDSchemeChoice.SUPERBEE.value == "Superbee"
    assert TVDSchemeChoice.MINMOD.value == "Minmod"
    assert TVDSchemeChoice.LIMITED_LINEAR.value == "LimitedLinear"
    assert TVDSchemeChoice.VAN_LEER.value == "VanLeer"
    assert TVDSchemeChoice.VAN_ALBADA.value == "VanAlbada"
    assert TVDSchemeChoice.UPWIND.value == "Upwind"


def test_tvd_factory_create_all_schemes():
    """Test TVDFactory creates all registered schemes."""
    schemes = [
        TVDSchemeChoice.SUPERBEE,
        TVDSchemeChoice.MINMOD,
        TVDSchemeChoice.LIMITED_LINEAR,
        TVDSchemeChoice.VAN_LEER,
        TVDSchemeChoice.VAN_ALBADA,
        TVDSchemeChoice.UPWIND,
    ]
    for choice in schemes:
        scheme = TVDFactory.create(choice)
        assert isinstance(scheme, ITVDScheme)


def test_tvd_factory_create_superbee():
    """Test TVDFactory creates Superbee scheme."""
    scheme = TVDFactory.create(TVDSchemeChoice.SUPERBEE)
    assert isinstance(scheme, Superbee)


def test_tvd_factory_create_minmod():
    """Test TVDFactory creates Minmod scheme."""
    scheme = TVDFactory.create(TVDSchemeChoice.MINMOD)
    assert isinstance(scheme, Minmod)


def test_tvd_factory_create_limited_linear():
    """Test TVDFactory creates LimitedLinear scheme."""
    scheme = TVDFactory.create(TVDSchemeChoice.LIMITED_LINEAR)
    assert isinstance(scheme, LimitedLinear)


def test_tvd_factory_create_van_leer():
    """Test TVDFactory creates VanLeer scheme."""
    scheme = TVDFactory.create(TVDSchemeChoice.VAN_LEER)
    assert isinstance(scheme, VanLeer)


def test_tvd_factory_create_van_albada():
    """Test TVDFactory creates VanAlbada scheme."""
    scheme = TVDFactory.create(TVDSchemeChoice.VAN_ALBADA)
    assert isinstance(scheme, VanAlbada)


def test_tvd_factory_create_upwind():
    """Test TVDFactory creates Upwind scheme."""
    scheme = TVDFactory.create(TVDSchemeChoice.UPWIND)
    assert isinstance(scheme, Upwind)


def test_tvd_factory_create_unknown():
    """Test TVDFactory raises error for unknown scheme."""
    mock_choice = Mock(spec=TVDSchemeChoice)
    mock_choice.name = "UNKNOWN"
    with pytest.raises(ValueError, match="Unknown TVD scheme choice"):
        TVDFactory.create(mock_choice)


def test_tvd_factory_register(monkeypatch):
    """Test TVDFactory register method."""

    class MockScheme(ITVDScheme):
        def correction_term(
            self, delta_minus: torch.Tensor, delta_plus: torch.Tensor
        ) -> torch.Tensor:
            return torch.zeros_like(delta_minus)

    # Use monkeypatch to safely modify registry
    original_registry = TVDFactory.registry.copy()
    mock_choice = TVDSchemeChoice.UPWIND
    mock_scheme = MockScheme()

    monkeypatch.setitem(TVDFactory.registry, mock_choice, mock_scheme)
    created = TVDFactory.create(mock_choice)
    assert isinstance(created, MockScheme)

    # Registry is automatically restored by monkeypatch
