import pytest
import pyvista as pv

from gridfoam._geometry import AABB, TriangleMesh


@pytest.fixture
def simple_mesh() -> TriangleMesh:
    """Create a simple mesh for testing."""
    mesh = pv.read("tests/data/vtu/simple/mesh.vtu").extract_surface()
    return TriangleMesh.from_polydata(mesh)


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


def test_per_face_aabbs(simple_mesh: TriangleMesh):
    """Test get_per_face_aabbs."""
    aabbs = simple_mesh.per_face_aabbs
    assert len(aabbs) == simple_mesh.n_triangles
    assert all(isinstance(aabb, AABB) for aabb in aabbs)


@pytest.mark.parametrize(
    "face_ids, expected_radius",
    [
        ([0, 1, 2, 3, 4, 5], 4.0),
        ([1, 2, 3, 4, 5, 6], 4.0),
    ],
)
def test_get_radii_of_curvature(
    sphere_mesh: TriangleMesh, face_ids: list[int], expected_radius: float
):
    """Test get_radii_of_curvature."""
    actual_radius = sphere_mesh.get_radii_of_curvature(face_ids)
    assert abs(actual_radius - expected_radius) < 0.1
