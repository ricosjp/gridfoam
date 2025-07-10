import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

from gridfoam._geometry import Plane


@pytest.fixture
def simple_plane() -> Plane:
    """Create a simple plane for testing."""
    normal = torch.tensor([1.0, 0.0, 0.0])
    distance = 1.0
    return Plane.from_normal_and_distance(normal, distance)


# property
@pytest.mark.parametrize(
    "normal, distance, expected_dim",
    [
        (torch.tensor([1.0, 0.0, 0.0]), 1.0, 3),
        (torch.tensor([1.0, 0.0]), 1.0, 2),
    ],
)
def test_plane_properties(
    normal: torch.Tensor, distance: float, expected_dim: int
):
    """Test Plane properties."""
    plane = Plane.from_normal_and_distance(normal, distance)
    assert plane.space_dim == expected_dim
    assert torch.equal(plane.normal, normal)
    assert plane.distance == distance


# classmethod
def test_plane_from_normal_and_distance():
    """Test Plane creation from normal and distance."""
    normal = torch.tensor([1.0, 0.0, 0.0])
    distance = 1.0
    plane = Plane.from_normal_and_distance(normal, distance)
    assert torch.equal(plane.normal, normal)
    assert plane.distance == distance


def test_plane_from_normal_and_point():
    """Test Plane creation from normal and point."""
    input_normal = torch.tensor([2.0, 0.0, 0.0])
    expected_normal = torch.tensor([1.0, 0.0, 0.0])
    point = torch.tensor([1.0, 2.0, 0.0])
    plane = Plane.from_normal_and_point(input_normal, point)
    assert torch.equal(plane.normal, expected_normal)
    assert plane.distance == 1.0


def test_invalid_input():
    """Test invalid input."""
    # dimension mismatch
    with pytest.raises(BeartypeCallHintParamViolation):
        Plane.from_normal_and_distance(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            1.0,
        )
    with pytest.raises(BeartypeCallHintParamViolation):
        Plane.from_normal_and_point(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([1.0, 0.0, 0.0]),
        )
    with pytest.raises(BeartypeCallHintParamViolation):
        Plane.from_normal_and_point(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
    # shape mismatch
    with pytest.raises(RuntimeError):
        Plane.from_normal_and_point(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0]),
        )
    # zero normal
    with pytest.raises(ValueError):
        Plane.from_normal_and_point(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        )
