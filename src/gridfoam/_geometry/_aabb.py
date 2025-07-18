from __future__ import annotations

from itertools import product

import torch
from jaxtyping import Float


class AABB:
    """
    Axis-Aligned Bounding Box
    """

    def __init__(
        self,
        min_pt: Float[torch.Tensor, " space_dim"],
        max_pt: Float[torch.Tensor, " space_dim"],
    ):
        """Create an AABB from minimum and maximum points.

        Parameters
        ----------
        min_pt : Float[torch.Tensor, " space_dim"]
            minimum point of the AABB
        max_pt : Float[torch.Tensor, " space_dim"]
            maximum point of the AABB
        """
        if min_pt.shape[0] != max_pt.shape[0]:
            raise ValueError(
                "center and halfwidth must have the same dimension"
            )
        if torch.any(max_pt < min_pt):
            raise ValueError("max_pt must be greater than min_pt")
        self._space_dim = min_pt.shape[0]
        self._min_pt = min_pt
        self._max_pt = max_pt

    @property
    def space_dim(self) -> int:
        return self._space_dim

    @property
    def center(self) -> Float[torch.Tensor, " space_dim"]:
        return self._min_pt + self.halfwidth

    @property
    def halfwidth(self) -> Float[torch.Tensor, " space_dim"]:
        return self.width * 0.5

    @property
    def width(self) -> Float[torch.Tensor, " space_dim"]:
        return self.max - self.min

    @property
    def min(self) -> Float[torch.Tensor, " space_dim"]:
        return self._min_pt

    @property
    def max(self) -> Float[torch.Tensor, " space_dim"]:
        return self._max_pt

    @property
    def bounds(self) -> Float[torch.Tensor, " space_dimx2"]:
        return torch.hstack([self.min, self.max])

    @classmethod
    def from_points(
        cls, pts: Float[torch.Tensor, "n_points space_dim"]
    ) -> AABB:
        """
        Create an minimal AABB such that all points are inside.

        Parameters
        ----------
        pts : Float[torch.Tensor, "n_points space_dim"]
            points to be enclosed

        Returns
        -------
        AABB
            AABB object
        """
        min_pt = torch.min(pts, dim=0).values
        max_pt = torch.max(pts, dim=0).values
        return cls(min_pt, max_pt)

    @classmethod
    def get_per_triangle_aabbs(
        cls, triangles: Float[torch.Tensor, "n_triangles 3 space_dim"]
    ) -> list[AABB]:
        """
        Create an list of AABBs such that each triangle is inside.

        Parameters
        ----------
        triangles : Float[torch.Tensor, "n_triangles 3 space_dim"]
            triangles to be enclosed

        Returns
        -------
        list[AABB]
            list of AABBs
        """
        return [cls.from_points(triangle) for triangle in triangles]

    def contains(self, pt: Float[torch.Tensor, " space_dim"]) -> bool:
        """
        Check if a point is inside the AABB.
        """
        cond = (self.min <= pt) & (pt <= self.max)
        return torch.all(cond).item()

    def split(self) -> list[AABB]:
        """
        Split the AABB into 2^space_dim AABBs.

        Returns
        -------
        list[AABB]
            list of AABBs. The order of the AABBs is Morton-order.
        """
        center = self.min + (self.max - self.min) * 0.5
        children = []
        bits = torch.tensor(
            [tuple(reversed(p)) for p in product([0, 1], repeat=self.space_dim)]
        )
        mins = torch.where(bits == 0, self.min, center)
        maxs = torch.where(bits == 0, center, self.max)
        children = [
            AABB(min_pt, max_pt)
            for min_pt, max_pt in zip(mins, maxs, strict=True)
        ]
        return children

    def merge(self, other: AABB) -> AABB:
        """
        Merge two AABBs.

        Parameters
        ----------
        other : AABB
            other AABB

        Returns
        -------
        AABB
            merged AABB
        """
        min_pt = torch.min(self.min, other.min)
        max_pt = torch.max(self.max, other.max)
        return AABB(min_pt, max_pt)
