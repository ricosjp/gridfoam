"""Tests for constants module."""

import pytest

from gridfoam.DNA.constants import DOMAIN_BOUNDARY_MAP, FACE_NEIGHBOR_MAP
from gridfoam.DNA.enum import Axis


def test_face_neighbor_map():
    """Test FACE_NEIGHBOR_MAP."""
    assert FACE_NEIGHBOR_MAP[4] == (Axis.Z, False)
    assert FACE_NEIGHBOR_MAP[10] == (Axis.Y, False)
    assert FACE_NEIGHBOR_MAP[12] == (Axis.X, False)
    assert FACE_NEIGHBOR_MAP[14] == (Axis.X, True)
    assert FACE_NEIGHBOR_MAP[16] == (Axis.Y, True)
    assert FACE_NEIGHBOR_MAP[22] == (Axis.Z, True)


def test_domain_boundary_map():
    """Test DOMAIN_BOUNDARY_MAP."""
    assert DOMAIN_BOUNDARY_MAP["domainZ-"] == 4
    assert DOMAIN_BOUNDARY_MAP["domainY-"] == 10
    assert DOMAIN_BOUNDARY_MAP["domainX-"] == 12
    assert DOMAIN_BOUNDARY_MAP["domainX+"] == 14
    assert DOMAIN_BOUNDARY_MAP["domainY+"] == 16
    assert DOMAIN_BOUNDARY_MAP["domainZ+"] == 22
