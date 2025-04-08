import pytest
import torch

from gridfoam._base import GridTensor, grid_tensor
from gridfoam._structure.octree.aabb import AABB
from gridfoam.utils.enums import CoordinateType


@pytest.fixture
def simple_aabb() -> AABB:
    """Create a simple AABB for testing."""
    min_point = grid_tensor([0.0, 0.0, 0.0])
    max_point = grid_tensor([1.0, 1.0, 1.0])
    return AABB.from_min_and_max(min_point, max_point)


def test_aabb_default_constructor():
    """Test AABB default constructor."""
    aabb = AABB(3)
    assert aabb.min.shape == (3,)
    assert aabb.max.shape == (3,)
    assert torch.equal(aabb.min, torch.full((3,), torch.inf))
    assert torch.equal(aabb.max, torch.full((3,), -torch.inf))


# classmethod
def test_aabb_from_min_and_max():
    """Test AABB creation from min and max points."""
    min_point = grid_tensor([0.0, 0.0, 0.0])
    max_point = grid_tensor([1.0, 1.0, 1.0])
    aabb = AABB.from_min_and_max(min_point, max_point)
    assert aabb.min.shape == (3,)
    assert aabb.max.shape == (3,)
    assert torch.equal(aabb.min, min_point)
    assert torch.equal(aabb.max, max_point)


def test_aabb_from_origin_and_size():
    """Test AABB creation from origin and size."""
    origin = grid_tensor([0.0, 0.0, 0.0])
    size = grid_tensor([1.0, 1.0, 1.0])
    aabb = AABB.from_origin_and_size(origin, size)
    assert aabb.min.shape == (3,)
    assert aabb.max.shape == (3,)
    assert torch.equal(aabb.min, origin)
    assert torch.equal(aabb.max, origin + size)


def test_aabb_from_points():
    """Test AABB creation from points."""
    points = grid_tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    aabb = AABB.from_points(points)
    assert aabb.min.shape == (3,)
    assert aabb.max.shape == (3,)
    assert torch.equal(aabb.min, points[0])
    assert torch.equal(aabb.max, points[1])


def test_aabb_from_triangles():
    """Test AABB creation from triangles."""
    triangles = grid_tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]],
        ]
    )
    expected_mins = [
        grid_tensor([0.0, 0.0, 0.0]),
        grid_tensor([1.0, 1.0, 1.0]),
    ]
    expected_maxs = [
        grid_tensor([1.0, 1.0, 0.0]),
        grid_tensor([2.0, 2.0, 1.0]),
    ]
    aabbs = AABB.from_triangles(triangles)
    for aabb, expected_min, expected_max in zip(
        aabbs, expected_mins, expected_maxs, strict=False
    ):
        assert aabb.min.shape == (3,)
        assert aabb.max.shape == (3,)
        assert torch.equal(aabb.min, expected_min)
        assert torch.equal(aabb.max, expected_max)


def test_aabb_from_aabbs():
    """Test AABB creation from a list of AABBs."""
    aabbs = [
        AABB.from_min_and_max(
            grid_tensor([0.0, 0.0, 0.0]), grid_tensor([1.0, 1.0, 1.0])
        ),
        AABB.from_min_and_max(
            grid_tensor([1.0, 1.0, 1.0]), grid_tensor([2.0, 2.0, 2.0])
        ),
    ]
    aabb = AABB.from_aabbs(aabbs)
    assert aabb.min.shape == (3,)
    assert aabb.max.shape == (3,)
    assert torch.equal(aabb.min, grid_tensor([0.0, 0.0, 0.0]))
    assert torch.equal(aabb.max, grid_tensor([2.0, 2.0, 2.0]))


@pytest.mark.parametrize(
    "point, should_contain",
    [
        (GridTensor(torch.tensor([0.5, 0.5, 0.5])), True),
        (GridTensor(torch.tensor([1.5, 0.5, 0.5])), False),
        (GridTensor(torch.tensor([0.5, 1.5, 0.5])), False),
        (GridTensor(torch.tensor([0.5, 0.5, 1.5])), False),
    ],
)
def test_aabb_contains(
    simple_aabb: AABB, point: GridTensor, should_contain: bool
):
    """Test point containment in AABB."""
    assert (point in simple_aabb) == should_contain


def test_aabb_push():
    """Test AABB push."""
    aabb = AABB.from_min_and_max(
        grid_tensor([0.0, 0.0, 0.0]), grid_tensor([1.0, 1.0, 1.0])
    )
    aabb.push(grid_tensor([0.5, 0.5, 0.5]))
    assert torch.equal(aabb.min, grid_tensor([0.0, 0.0, 0.0]))
    assert torch.equal(aabb.max, grid_tensor([1.0, 1.0, 1.0]))
    aabb.push(grid_tensor([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]))
    assert torch.equal(aabb.min, grid_tensor([0.0, 0.0, 0.0]))
    assert torch.equal(aabb.max, grid_tensor([1.5, 1.5, 1.5]))


def test_aabb_scale():
    """Test AABB scale."""
    aabb = AABB.from_min_and_max(
        grid_tensor([0.0, 0.0, 0.0]), grid_tensor([1.0, 1.0, 1.0])
    )
    aabb.scale(2.0)
    assert torch.equal(aabb.min, grid_tensor([-0.5, -0.5, -0.5]))
    assert torch.equal(aabb.max, grid_tensor([1.5, 1.5, 1.5]))


def test_aabb_center():
    """Test AABB center."""
    aabb = AABB.from_min_and_max(
        grid_tensor([0.0, 0.0, 0.0]), grid_tensor([1.0, 1.0, 1.0])
    )
    assert torch.equal(aabb.center(), grid_tensor([0.5, 0.5, 0.5]))


def test_aabb_delta():
    """Test AABB delta."""
    aabb = AABB.from_min_and_max(
        grid_tensor([-1.0, 0.0, 1.0]), grid_tensor([2.0, 2.0, 2.0])
    )
    assert torch.equal(aabb.delta(), grid_tensor([3.0, 2.0, 1.0]))


def test_aabb_subdivide():
    """Test AABB subdivision."""
    aabb = AABB.from_min_and_max(
        grid_tensor([0.0, 0.0, 0.0], CoordinateType.GLOBAL),
        grid_tensor([1.0, 1.0, 1.0], CoordinateType.GLOBAL),
    )
    sub_aabbs = aabb.subdivide()
    assert len(sub_aabbs) == 8
    expected_centers = [
        grid_tensor([0.25, 0.25, 0.25]),
        grid_tensor([0.75, 0.25, 0.25]),
        grid_tensor([0.25, 0.75, 0.25]),
        grid_tensor([0.75, 0.75, 0.25]),
        grid_tensor([0.25, 0.25, 0.75]),
        grid_tensor([0.75, 0.25, 0.75]),
        grid_tensor([0.25, 0.75, 0.75]),
        grid_tensor([0.75, 0.75, 0.75]),
    ]
    for aabb, expected_center in zip(sub_aabbs, expected_centers, strict=False):
        assert torch.equal(aabb.center(), expected_center)
        assert torch.equal(aabb.delta(), grid_tensor([0.5, 0.5, 0.5]))


@pytest.mark.parametrize(
    "other_aabb, should_intersect",
    [
        (
            AABB.from_min_and_max(
                GridTensor(torch.tensor([0.5, 0.5, 0.5])),
                GridTensor(torch.tensor([1.5, 1.5, 1.5])),
            ),
            True,
        ),
        (
            AABB.from_min_and_max(
                GridTensor(torch.tensor([2.0, 2.0, 2.0])),
                GridTensor(torch.tensor([3.0, 3.0, 3.0])),
            ),
            False,
        ),
    ],
)
def test_aabb_intersect(
    simple_aabb: AABB, other_aabb: AABB, should_intersect: bool
):
    """Test AABB intersection."""
    assert simple_aabb.intersect_aabb(other_aabb) == should_intersect


def test_invalid_input():
    """Test invalid input."""
    with pytest.raises(ValueError):
        AABB.from_min_and_max(
            grid_tensor(torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
            grid_tensor(torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])),
        )
    with pytest.raises(ValueError):
        AABB.from_min_and_max(
            grid_tensor(torch.tensor([0.0, 0.0, 0.0])),
            grid_tensor(torch.tensor([1.0, 1.0, 1.0, 1.0])),
        )
    with pytest.raises(ValueError):
        AABB.from_min_and_max(
            grid_tensor(torch.tensor([0.0, 0.0, 0.0])),
            grid_tensor(torch.tensor([-1.0, -1.0, -1.0])),
        )

    with pytest.raises(ValueError):
        AABB.from_origin_and_size(
            grid_tensor(torch.tensor([0.0, 0.0, 0.0])),
            grid_tensor(torch.tensor([-1.0, 1.0, 1.0])),
        )

    with pytest.raises(ValueError):
        AABB.from_points(
            grid_tensor(torch.tensor([0.0, 0.0, 0.0])),
        )

    with pytest.raises(ValueError):
        AABB.from_triangles(
            grid_tensor(
                torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
                )
            ),
        )

    with pytest.raises(ValueError):
        aabb = AABB.from_min_and_max(
            grid_tensor(torch.tensor([0.0, 0.0, 0.0])),
            grid_tensor(torch.tensor([1.0, 1.0, 1.0])),
        )
        tensor = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        _ = tensor in aabb
    with pytest.raises(ValueError):
        aabb = AABB.from_min_and_max(
            grid_tensor(torch.tensor([0.0, 0.0, 0.0])),
            grid_tensor(torch.tensor([1.0, 1.0, 1.0])),
        )
        tensor = torch.tensor(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
            ]
        )
        aabb.push(tensor)
