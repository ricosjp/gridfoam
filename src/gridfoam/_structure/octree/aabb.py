from __future__ import annotations

import torch

from gridfoam._base import GridTensor, grid_tensor
from gridfoam.utils.enums import CoordinateType


class AABB:
    """Axis-Aligned Bounding Box"""

    def __init__(self, dim: int):
        self.min = GridTensor.full((dim,), torch.inf)
        self.max = GridTensor.full((dim,), -torch.inf)

    @classmethod
    def from_min_and_max(cls, min_pt: GridTensor, max_pt: GridTensor) -> AABB:
        if min_pt.ndim != 1 or max_pt.ndim != 1:
            raise ValueError("min and max must be 1D tensors")
        if min_pt.shape != max_pt.shape:
            raise ValueError("min and max must be same shape")
        if torch.any(min_pt > max_pt):
            raise ValueError("min must be less than max")
        dim = min_pt.shape[0]
        aabb = cls(dim)
        aabb.min = min_pt
        aabb.max = max_pt
        return aabb

    @classmethod
    def from_origin_and_size(cls, orig: GridTensor, size: GridTensor) -> AABB:
        if torch.any(size < 0.0):
            raise ValueError("size must be non-negative")
        return cls.from_min_and_max(orig, orig + size)

    @classmethod
    def from_points(cls, pts: GridTensor) -> AABB:
        if pts.ndim != 2:
            raise ValueError("points must be a 2D tensor")
        coord_type = pts.coord_type
        min_pt = grid_tensor(torch.min(pts, dim=0).values, coord_type)
        max_pt = grid_tensor(torch.max(pts, dim=0).values, coord_type)
        return cls.from_min_and_max(min_pt, max_pt)

    @classmethod
    def from_triangles(cls, triangles: GridTensor) -> list[AABB]:
        if triangles.ndim != 3:
            raise ValueError("triangles must be a 3D tensor")
        return [cls.from_points(triangle) for triangle in triangles]

    @classmethod
    def from_aabbs(cls, aabbs: list[AABB]) -> AABB:
        coord_type = aabbs[0].min.coord_type
        min_pt = grid_tensor(
            torch.min(torch.stack([aabb.min for aabb in aabbs]), dim=0).values,
            coord_type,
        )
        max_pt = grid_tensor(
            torch.max(torch.stack([aabb.max for aabb in aabbs]), dim=0).values,
            coord_type,
        )
        return cls.from_min_and_max(min_pt, max_pt)

    def __contains__(self, pt: GridTensor) -> bool:
        if pt.shape != self.min.shape:
            raise ValueError("point must be a same shape with min or max")
        return torch.all(self.min <= pt) and torch.all(self.max >= pt)

    def push(self, pts: GridTensor):
        if pts.ndim > 2:
            raise ValueError("points must be a 1D or 2D tensor")
        coord_type = pts.coord_type
        self.min = grid_tensor(
            torch.min(torch.vstack([self.min, pts]), dim=0).values,
            coord_type,
        )
        self.max = grid_tensor(
            torch.max(torch.vstack([self.max, pts]), dim=0).values,
            coord_type,
        )

    def scale(self, factor: float):
        center = self.center()
        center_orig_min = self.min - center
        center_orig_max = self.max - center
        center_orig_min *= factor
        center_orig_max *= factor
        self.min = center_orig_min + center
        self.max = center_orig_max + center

    def center(self) -> GridTensor:
        return (self.min + self.max) / 2.0

    def delta(self) -> GridTensor:
        return self.max - self.min

    def subdivide(self, resolutions: GridTensor | None = None) -> list[AABB]:
        if resolutions is None:
            resolutions = grid_tensor([2, 2, 2], CoordinateType.CELL_INDEX)
        if resolutions.shape != self.min.shape:
            raise ValueError("resolutions must be a same shape with min or max")
        coord_type = self.min.coord_type
        origin = self.min
        offset = self.delta() / resolutions.to(coord_type=coord_type)

        X = torch.arange(resolutions[0].item())
        Y = torch.arange(resolutions[1].item())
        Z = torch.arange(resolutions[2].item())
        ZZ, YY, XX = torch.meshgrid(Z, Y, X, indexing="ij")
        shifts = grid_tensor(
            torch.stack([XX, YY, ZZ], dim=-1).reshape(-1, 3), coord_type
        )
        deltas = shifts * offset
        new_origins = origin + deltas
        return [AABB.from_origin_and_size(orig, offset) for orig in new_origins]

    def intersect_aabb(self, other: AABB) -> bool:
        return torch.all(self.min <= other.max) and torch.all(
            self.max >= other.min
        )

    def intersect_triangle(self, triangles: GridTensor) -> torch.Tensor:
        if triangles.ndim != 3:
            raise ValueError("triangles must be a 3D tensor")
        mins = torch.min(triangles, dim=1).values
        maxs = torch.max(triangles, dim=1).values
        c1 = torch.all(torch.unsqueeze(self.min, 0) <= maxs, dim=1)
        c2 = torch.all(torch.unsqueeze(self.max, 0) >= mins, dim=1)
        return torch.logical_and(c1, c2)
