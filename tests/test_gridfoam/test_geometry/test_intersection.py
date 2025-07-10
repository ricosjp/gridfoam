import pytest
import torch

from gridfoam._geometry import AABB, Plane
from gridfoam._geometry._intersection import (
    is_intersect_aabb_aabb,
    is_intersect_aabb_plane,
    is_intersect_aabb_triangle,
)


@pytest.mark.parametrize(
    "aabb1_center,aabb1_halfwidth,aabb2_center,aabb2_halfwidth,expected",
    [
        # Case: completely overlapping
        (
            torch.zeros(3),
            torch.ones(3),
            torch.zeros(3),
            torch.ones(3),
            True,
        ),
        # Case: partially overlapping
        (
            torch.zeros(3),
            torch.ones(3),
            torch.ones(3),
            torch.ones(3),
            True,
        ),
        # Case: just touching
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor([2.0, 0.0, 0.0]),
            torch.ones(3),
            True,
        ),
        # Case: separated
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor([3.0, 0.0, 0.0]),
            torch.ones(3),
            False,
        ),
    ],
)
def test_is_intersect_aabb_aabb(
    aabb1_center: torch.Tensor,
    aabb1_halfwidth: torch.Tensor,
    aabb2_center: torch.Tensor,
    aabb2_halfwidth: torch.Tensor,
    expected: bool,
):
    """Test AABB-AABB intersection detection."""
    aabb1 = AABB(center=aabb1_center, halfwidth=aabb1_halfwidth)
    aabb2 = AABB(center=aabb2_center, halfwidth=aabb2_halfwidth)
    assert is_intersect_aabb_aabb(aabb1, aabb2) == expected


@pytest.mark.parametrize(
    "aabb_center,aabb_halfwidth,plane_normal,plane_distance,expected",
    [
        # Case: plane completely crosses the AABB
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor([1.0, 0.0, 0.0]),
            0.0,
            True,
        ),
        # Case: plane touches the AABB
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor([1.0, 0.0, 0.0]),
            1.0,
            True,
        ),
        # Case: plane is separated from the AABB
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor([1.0, 0.0, 0.0]),
            2.0,
            False,
        ),
        # Case: diagonal plane intersects the AABB
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor([1.0, 1.0, 1.0]),
            0.0,
            True,
        ),
    ],
)
def test_is_intersect_aabb_plane(
    aabb_center: torch.Tensor,
    aabb_halfwidth: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_distance: float,
    expected: bool,
):
    """Test AABB-Plane intersection detection."""
    aabb = AABB(center=aabb_center, halfwidth=aabb_halfwidth)
    plane = Plane(normal=plane_normal, distance=plane_distance)
    assert is_intersect_aabb_plane(aabb, plane) == expected


@pytest.mark.parametrize(
    "aabb_center,aabb_halfwidth,triangle_vertices,expected",
    [
        # Case: intersecting
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                ]
            ),
            True,
        ),
        # Case: triangle touches AABB at a vertex
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            True,
        ),
        # Case: triangle touches AABB at a vertex
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [3.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 3.0],
                ]
            ),
            True,
        ),
        # Case: triangle touches AABB at an edge
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            True,
        ),
        # Case: triangle touches AABB at a face
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                ]
            ),
            True,
        ),
        # Case: triangle is separated from AABB
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [4.0, 0.0, 0.0],
                    [0.0, 4.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]
            ),
            False,
        ),
        # Case: triangle intersects AABB at an edge
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                ]
            ),
            True,
        ),
        # Case: triangle is completely contained in AABB
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                ]
            ),
            True,
        ),
        # Case: triangle is outside the AABB
        (
            torch.zeros(3),
            torch.ones(3),
            torch.tensor(
                [
                    [2.0, 0.0, 0.5],
                    [4.0, 0.0, 0.0],
                    [2.0, 3.0, 0.0],
                ]
            ),
            False,
        ),
    ],
)
def test_is_intersect_aabb_triangle(
    aabb_center: torch.Tensor,
    aabb_halfwidth: torch.Tensor,
    triangle_vertices: torch.Tensor,
    expected: bool,
):
    """Test AABB-Triangle intersection detection."""
    aabb = AABB(center=aabb_center, halfwidth=aabb_halfwidth)
    assert is_intersect_aabb_triangle(aabb, triangle_vertices) == expected
