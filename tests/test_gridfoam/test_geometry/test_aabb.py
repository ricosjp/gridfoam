import numpy as np
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from gridfoam._geometry import AABB


@pytest.fixture
def simple_aabb() -> AABB:
    """Create a simple AABB for testing."""
    min_pt = torch.tensor([0.0, 0.0, 0.0])
    max_pt = torch.tensor([1.0, 1.0, 1.0])
    return AABB(min_pt, max_pt)


# property
@pytest.mark.parametrize(
    "min_pt, max_pt, expected_dim",
    [
        (torch.tensor([-0.5, -0.5, -0.5]), torch.tensor([0.5, 0.5, 0.5]), 3),
        (torch.tensor([-0.5, -0.5]), torch.tensor([0.5, 0.5]), 2),
    ],
)
def test_aabb_properties(
    min_pt: torch.Tensor, max_pt: torch.Tensor, expected_dim: int
):
    """Test AABB properties."""
    aabb = AABB(min_pt, max_pt)
    assert aabb.space_dim == expected_dim
    assert torch.equal(aabb.min, min_pt)
    assert torch.equal(aabb.max, max_pt)
    assert torch.equal(aabb.center, (min_pt + max_pt) / 2.0)
    assert torch.equal(aabb.halfwidth, (max_pt - min_pt) / 2.0)
    assert torch.equal(aabb.width, max_pt - min_pt)
    assert torch.equal(aabb.bounds, torch.hstack([min_pt, max_pt]))


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


def test_aabb_merge(simple_aabb: AABB):
    """Test AABB merge."""
    other_aabb = AABB(
        torch.tensor([1.0, 1.0, 1.0]), torch.tensor([2.0, 2.0, 2.0])
    )
    merged_aabb = simple_aabb.merge(other_aabb)
    assert torch.equal(merged_aabb.min, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(merged_aabb.max, torch.tensor([2.0, 2.0, 2.0]))


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
    # negative width
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
