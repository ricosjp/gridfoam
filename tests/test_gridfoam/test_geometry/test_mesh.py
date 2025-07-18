import pathlib

import pytest
import pyvista as pv
import torch

from gridfoam._geometry import AABB, TriangleMesh


@pytest.fixture
def simple_mesh() -> TriangleMesh:
    """Create a simple mesh for testing."""
    path = pathlib.Path("tests/data/vtu/simple/mesh.vtu")
    return TriangleMesh.from_file(path)


@pytest.fixture
def sphere_mesh() -> TriangleMesh:
    """Create a sphere mesh for testing."""
    mesh = pv.Sphere(radius=2.0)
    return TriangleMesh.from_polydata(mesh)


def test_mesh_properties(simple_mesh: TriangleMesh):
    """Test mesh properties."""
    assert simple_mesh.space_dim == 3
    assert simple_mesh.n_points == 12
    assert simple_mesh.n_triangles == 20


def test_find_intersecting_face_ids(simple_mesh: TriangleMesh):
    """Test get_per_face_aabbs."""
    face_ids = simple_mesh.find_intersecting_face_ids(AABB(
        min_pt=torch.tensor([0.0, 0.0, 0.0]),
        max_pt=torch.tensor([1.0, 1.0, 1.0]),
    ))
    assert len(face_ids) == simple_mesh.n_triangles


@pytest.mark.parametrize(
    "face_ids, expected_radius",
    [
        ([0, 1, 2, 3, 4, 5], 2.0),
        ([1, 2, 3, 4, 5, 6], 2.0),
    ],
)
def test_get_radii_of_curvature(
    sphere_mesh: TriangleMesh, face_ids: list[int], expected_radius: float
):
    """Test get_radii_of_curvature."""
    actual_radius = sphere_mesh.calculate_radii_of_curvature(face_ids)
    assert abs(actual_radius - expected_radius) < 0.1
