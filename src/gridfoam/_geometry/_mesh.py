from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pyvista as pv
import torch
from jaxtyping import Float, Int

from gridfoam._geometry import AABB


@dataclass
class TriangleMesh:
    points: Float[torch.Tensor, "n_points 3"]
    faces: Int[torch.Tensor, "n_triangles 3"]
    gaussian_curvatures: Float[torch.Tensor, " n_points"]

    @classmethod
    def from_polydata(cls, polydata: pv.PolyData) -> TriangleMesh:
        trimesh = polydata.triangulate()
        gaussian_curvatures = torch.from_numpy(polydata.curvature("gaussian"))
        faces = torch.from_numpy(trimesh.regular_faces)
        points = torch.from_numpy(trimesh.points)
        return cls(points, faces, gaussian_curvatures)

    @cached_property
    def per_face_aabbs(self) -> list[AABB]:
        triangles = self.points[self.faces]
        return AABB.get_per_triangle_aabbs(triangles)

    def get_radii_of_curvature(self, face_ids: list[int]) -> float:
        if len(face_ids) == 0:
            return float("inf")
        point_ids = torch.unique(torch.flatten(self.faces[face_ids]))
        regional_curvature = self.gaussian_curvatures[point_ids]
        return 1.0 / torch.abs(regional_curvature).mean().item()

    @property
    def space_dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_triangles(self) -> int:
        return self.faces.shape[0]
