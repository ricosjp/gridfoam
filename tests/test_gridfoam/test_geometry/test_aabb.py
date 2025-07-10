import numpy as np
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from gridfoam._geometry import AABB


@pytest.fixture
def simple_aabb() -> AABB:
    """Create a simple AABB for testing."""
    center = torch.tensor([0.5, 0.5, 0.5])
    halfwidth = torch.tensor([0.5, 0.5, 0.5])
    return AABB(center, halfwidth)


# property
@pytest.mark.parametrize(
    "center, halfwidth, expected_dim",
    [
        (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0]), 3),
        (torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]), 2),
    ],
)
def test_aabb_properties(
    center: torch.Tensor, halfwidth: torch.Tensor, expected_dim: int
):
    """Test AABB properties."""
    aabb = AABB(center, halfwidth)
    assert aabb.space_dim == expected_dim
    assert torch.equal(aabb.center, center)
    assert torch.equal(aabb.halfwidth, halfwidth)
    assert torch.equal(aabb.width, halfwidth * 2.0)
    assert torch.equal(aabb.min, center - halfwidth)
    assert torch.equal(aabb.max, center + halfwidth)


# classmethod
def test_aabb_from_center_halfwidth():
    """Test AABB creation from center and halfwidth."""
    center = torch.tensor([0.5, 0.5, 0.5])
    halfwidth = torch.tensor([0.5, 0.5, 0.5])
    aabb = AABB.from_center_halfwidth(center, halfwidth)
    assert torch.equal(aabb.center, center)
    assert torch.equal(aabb.halfwidth, halfwidth)


def test_aabb_from_min_max():
    """Test AABB creation from min and max points."""
    min_point = torch.tensor([0.0, 0.0, 0.0])
    max_point = torch.tensor([1.0, 1.0, 1.0])
    aabb = AABB.from_min_max(min_point, max_point)
    assert torch.equal(aabb.min, min_point)
    assert torch.equal(aabb.max, max_point)
    assert torch.equal(aabb.center, (max_point + min_point) / 2.0)
    assert torch.equal(aabb.halfwidth, (max_point - min_point) / 2.0)


def test_aabb_from_origin_width():
    """Test AABB creation from origin and width."""
    origin = torch.tensor([0.0, 0.0, 0.0])
    width = torch.tensor([1.0, 1.0, 1.0])
    aabb = AABB.from_origin_width(origin, width)
    assert torch.equal(aabb.min, origin)
    assert torch.equal(aabb.max, origin + width)
    assert torch.equal(aabb.center, origin + width / 2.0)
    assert torch.equal(aabb.halfwidth, width / 2.0)


def test_aabb_from_points():
    """Test AABB creation from points."""
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    aabb = AABB.from_points(points)
    assert torch.equal(aabb.center, (points[0] + points[1]) / 2.0)
    assert torch.equal(aabb.halfwidth, (points[1] - points[0]) / 2.0)


def test_aabb_get_per_triangle_aabbs():
    """Test AABB creation from triangles."""
    triangles = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]],
        ]
    )
    expected_mins = [
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
    ]
    expected_maxs = [
        torch.tensor([1.0, 1.0, 0.0]),
        torch.tensor([2.0, 2.0, 1.0]),
    ]
    aabbs = AABB.get_per_triangle_aabbs(triangles)
    for aabb, expected_min, expected_max in zip(
        aabbs, expected_mins, expected_maxs, strict=True
    ):
        np.testing.assert_array_equal(aabb.min.numpy(), expected_min.numpy())
        np.testing.assert_array_equal(aabb.max.numpy(), expected_max.numpy())


@pytest.mark.parametrize(
    "point, should_contain",
    [
        (torch.tensor([0.5, 0.5, 0.5]), True),
        (torch.tensor([1.5, 0.5, 0.5]), False),
        (torch.tensor([0.5, 1.5, 0.5]), False),
        (torch.tensor([0.5, 0.5, 1.5]), False),
    ],
)
def test_aabb_contains(
    simple_aabb: AABB, point: torch.Tensor, should_contain: bool
):
    """Test point containment in AABB."""
    assert simple_aabb.contains(point) == should_contain


def test_aabb_scale(simple_aabb: AABB):
    """Test AABB scale."""
    simple_aabb.scale(2.0)
    assert torch.equal(simple_aabb.center, torch.tensor([0.5, 0.5, 0.5]))
    assert torch.equal(simple_aabb.halfwidth, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(simple_aabb.min, torch.tensor([-0.5, -0.5, -0.5]))
    assert torch.equal(simple_aabb.max, torch.tensor([1.5, 1.5, 1.5]))


def test_aabb_split(simple_aabb: AABB):
    """Test AABB subdivision."""
    sub_aabbs = simple_aabb.split()
    assert len(sub_aabbs) == 8
    expected_centers = [
        torch.tensor([0.25, 0.25, 0.25]),
        torch.tensor([0.75, 0.25, 0.25]),
        torch.tensor([0.25, 0.75, 0.25]),
        torch.tensor([0.75, 0.75, 0.25]),
        torch.tensor([0.25, 0.25, 0.75]),
        torch.tensor([0.75, 0.25, 0.75]),
        torch.tensor([0.25, 0.75, 0.75]),
        torch.tensor([0.75, 0.75, 0.75]),
    ]
    for aabb, expected_center in zip(sub_aabbs, expected_centers, strict=False):
        assert torch.equal(aabb.center, expected_center)
        assert torch.equal(aabb.halfwidth, torch.tensor([0.25, 0.25, 0.25]))


def test_invalid_input():
    """Test invalid input."""
    # dimension mismatch
    with pytest.raises(BeartypeCallHintParamViolation):
        AABB(
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
    with pytest.raises(BeartypeCallHintParamViolation):
        AABB(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        )
    # shape mismatch
    with pytest.raises(ValueError):
        AABB(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )
    # negative halfwidth
    with pytest.raises(ValueError):
        AABB(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([-1.0, -1.0, -1.0]),
        )
    # 1 point
    with pytest.raises(BeartypeCallHintParamViolation):
        AABB.from_points(
            torch.tensor([0.0, 0.0, 0.0]),
        )
    # 1 triangle
    with pytest.raises(BeartypeCallHintParamViolation):
        AABB.get_per_triangle_aabbs(
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        )
    # # different shape
    # with pytest.raises(BeartypeCallHintParamViolation):
    #     aabb = AABB(
    #         torch.tensor([0.0, 0.0, 0.0]),
    #         torch.tensor([1.0, 1.0, 1.0]),
    #     )
    #     tensor = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    #     aabb.contains(tensor)
