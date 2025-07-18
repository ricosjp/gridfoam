import torch
from jaxtyping import Float

from gridfoam._geometry._interface import IAABB, IPlane

########################################################
# References
# Real Time Collision Detection
# https://www.r-5.org/files/books/computers/algo-list/realtime-3d/Christer_Ericson-Real-Time_Collision_Detection-EN.pdf
########################################################


# AABB against AABB
def is_intersect_aabb_aabb(aabb1: IAABB, aabb2: IAABB) -> bool:
    d = aabb1.center - aabb2.center
    sum_halfwidth = aabb1.halfwidth + aabb2.halfwidth
    return torch.all(torch.abs(d) <= sum_halfwidth).item()


# AABB against Plane
def is_intersect_aabb_plane(aabb: IAABB, plane: IPlane) -> bool:
    c = aabb.center
    e = aabb.halfwidth
    # project halfwidth onto plane normal
    r = torch.sum(e * torch.abs(plane.normal))
    # project center onto plane normal
    s = torch.dot(plane.normal, c) - plane.distance
    return (torch.abs(s) <= r).item()


# AABB against Triangle
def is_intersect_aabb_triangle(
    aabb: IAABB, triangle: Float[torch.Tensor, "3 space_dim"]
) -> bool:
    """
    Determine whether an AABB intersects with a triangle.

    Parameters
    ----------
    aabb : IAABB
        The axis-aligned bounding box.
    triangle : Float[torch.Tensor, "3 space_dim"]
        The triangle vertices.

    Returns
    -------
    bool
        True if the AABB and triangle intersect, False otherwise.
    """
    # Translate triangle to AABB local space
    v = triangle - aabb.center  # (3, dim)

    # Compute triangle edges
    e0 = v[1] - v[0]  # (dim,)
    e1 = v[2] - v[1]  # (dim,)
    e2 = v[0] - v[2]  # (dim,)
    edges = torch.stack([e0, e1, e2])  # (3, dim)

    # Compute the triangle normal
    tri_normal = torch.linalg.cross(e0, e1)  # (dim,)
    if torch.all(tri_normal == 0):
        # Degenerate triangle
        return False
    tri_normal /= torch.norm(tri_normal)

    # Define AABB axes (unit vectors)
    aabb_axes = torch.eye(aabb.space_dim)  # (dim, dim)

    # Test the three AABB axes
    for i in range(aabb.space_dim):
        a = aabb_axes[i]  # (dim,)
        projs = v @ a  # (3,)
        p_min = projs.min()
        p_max = projs.max()
        r = aabb.halfwidth[i]
        if (p_min > r).item() or (p_max < -r).item():
            return False

    # Test the triangle normal axis
    d = tri_normal @ v[0]
    r = torch.sum(aabb.halfwidth * torch.abs(tri_normal))
    if torch.abs(d) > r:
        return False

    # Test the 9 cross product axes
    for i in range(aabb.space_dim):
        for j in range(edges.size(0)):
            axis = torch.linalg.cross(aabb_axes[i], edges[j])
            r = (aabb.halfwidth * torch.abs(axis)).sum()
            projs = v @ axis
            p_min = projs.min()
            p_max = projs.max()
            if (p_min > r).item() or (p_max < -r).item():
                return False

    # No separating axis found
    return True
