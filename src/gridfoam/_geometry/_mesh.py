from __future__ import annotations

import pathlib
from dataclasses import dataclass

import pyvista as pv
import rtree
import torch
from jaxtyping import Float, Int, Int32
from trimesh.triangles import bounds_tree

from gridfoam._geometry import AABB


@dataclass(frozen=True)
class TriangleMesh:
    points: Float[torch.Tensor, " n_points 3"]
    faces: Int[torch.Tensor, " n_triangles 3"]
    gaussian_curvatures: Float[torch.Tensor, " n_points"]
    tree: rtree.Rtree | None = None

    @classmethod
    def from_file(cls, file: pathlib.Path) -> TriangleMesh:
        polydata: pv.PolyData = pv.read(file).extract_surface()
        polydata.triangulate(inplace=True)
        points = torch.from_numpy(polydata.points)
        faces = torch.from_numpy(polydata.regular_faces)
        gaussian_curvatures = torch.from_numpy(polydata.curvature("gaussian"))
        tree = bounds_tree(points[faces])
        return cls(points, faces, gaussian_curvatures, tree)

    @classmethod
    def from_polydata(cls, polydata: pv.PolyData) -> TriangleMesh:
        tri = polydata.triangulate(inplace=True)
        points = torch.from_numpy(tri.points)
        faces = torch.from_numpy(tri.regular_faces)
        gaussian_curvatures = torch.from_numpy(tri.curvature("gaussian"))
        tree = bounds_tree(points[faces])
        return cls(points, faces, gaussian_curvatures, tree)

    def find_intersecting_face_ids(self, aabb: AABB) -> Int32[torch.Tensor, " n_faces"]:
        """Find the IDs of faces that intersect with the given AABB bounds.

        Parameters
        ----------
        aabb : AABB
            The axis-aligned bounding box.

        Returns
        -------
        Int32[torch.Tensor, " n_faces"]
            Intersecting face IDs.
        """
        if self.tree is None:
            return torch.tensor([], dtype=torch.int32)
        aabb_bounds = aabb.bounds
        hit_indices = list(self.tree.intersection(aabb_bounds.tolist()))
        return torch.tensor(hit_indices, dtype=torch.int32)

    def calculate_radii2_of_curvature(self, face_ids: Int32[torch.Tensor, " n_faces"]) -> float:
        if len(face_ids) == 0:
            return float("inf")
        point_ids = torch.unique(torch.flatten(self.faces[face_ids]))
        inv_r2 = torch.abs(self.gaussian_curvatures[point_ids])
        r2_g = 1.0 / inv_r2.mean().item()  # radius of effective curvature
        return r2_g

    @property
    def space_dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_triangles(self) -> int:
        return self.faces.shape[0]
