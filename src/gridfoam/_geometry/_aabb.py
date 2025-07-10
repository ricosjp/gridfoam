from __future__ import annotations

import itertools

import torch
from jaxtyping import Float

from gridfoam._geometry._interface import IAABB


class AABB(IAABB):
    """
    Axis-Aligned Bounding Box
    represented by center and halfwidth for memory and precision reasons.
    """

    def __init__(
        self,
        center: Float[torch.Tensor, " space_dim"],
        halfwidth: Float[torch.Tensor, " space_dim"],
    ):
        if center.shape[0] != halfwidth.shape[0]:
            raise ValueError(
                "center and halfwidth must have the same dimension"
            )
        if torch.any(halfwidth < 0.0):
            raise ValueError("halfwidth must be non-negative")
        self._space_dim = center.shape[0]
        self._center = center
        self._halfwidth = halfwidth

    @property
    def space_dim(self) -> int:
        return self._space_dim

    @property
    def center(self) -> Float[torch.Tensor, " space_dim"]:
        return self._center

    @property
    def halfwidth(self) -> Float[torch.Tensor, " space_dim"]:
        return self._halfwidth

    @property
    def width(self) -> Float[torch.Tensor, " space_dim"]:
        return self._halfwidth * 2.0

    @property
    def min(self) -> Float[torch.Tensor, " space_dim"]:
        """
        Minimum point of the AABB.

        Notes
        -----
        This property is not recommended for use
        due to potential loss of precision.
        Consider using center and halfwidth directly
        for calculations that require high accuracy.
        """
        return self._center - self._halfwidth

    @property
    def max(self) -> Float[torch.Tensor, " space_dim"]:
        """
        Maximum point of the AABB.

        Notes
        -----
        This property is not recommended for use
        due to potential loss of precision.
        Consider using center and halfwidth directly
        for calculations that require high accuracy.
        """
        return self._center + self._halfwidth

    @classmethod
    def from_center_halfwidth(
        cls,
        center: Float[torch.Tensor, " space_dim"],
        halfwidth: Float[torch.Tensor, " space_dim"],
    ) -> AABB:
        """
        Create an AABB from center and halfwidth.

        Parameters
        ----------
        center : Float[torch.Tensor, " space_dim"]
            center of the AABB
        halfwidth : Float[torch.Tensor, " space_dim"]
            halfwidth of the AABB

        Returns
        -------
        AABB
            AABB object
        """
        return cls(center, halfwidth)

    @classmethod
    def from_min_max(
        cls,
        min_pt: Float[torch.Tensor, " space_dim"],
        max_pt: Float[torch.Tensor, " space_dim"],
    ) -> AABB:
        """
        Create an AABB from minimum and maximum points.

        Parameters
        ----------
        min_pt : Float[torch.Tensor, " space_dim"]
            minimum point of the AABB
        max_pt : Float[torch.Tensor, " space_dim"]
            maximum point of the AABB

        Returns
        -------
        AABB
            AABB object

        Notes
        -----
        This method may cause precision issues.
        Consider using center and halfwidth directly
        for calculations that require high accuracy.
        """
        # this subtraction may cause precision issues
        halfwidth = (max_pt - min_pt) * 0.5
        center = min_pt + halfwidth
        return cls(center, halfwidth)

    @classmethod
    def from_origin_width(
        cls,
        orig: Float[torch.Tensor, " space_dim"],
        width: Float[torch.Tensor, " space_dim"],
    ) -> AABB:
        """
        Create an AABB from origin and width.

        Parameters
        ----------
        orig : Float[torch.Tensor, " space_dim"]
            origin of the AABB
        width : Float[torch.Tensor, " space_dim"]
            width of the AABB

        Returns
        -------
        AABB
            AABB object
        """
        halfwidth = width * 0.5
        center = orig + halfwidth
        return cls(center, halfwidth)

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
        base = pts[0]
        relative_points = pts - base
        min_rel = torch.min(relative_points, dim=0).values
        max_rel = torch.max(relative_points, dim=0).values
        center_rel = (min_rel + max_rel) * 0.5
        halfwidth = (max_rel - min_rel) * 0.5
        center = base + center_rel
        return cls(center, halfwidth)

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

    def contains(
        self, pt: Float[torch.Tensor, " space_dim"]
    ) -> bool:
        """
        Check if a point is inside the AABB.
        """
        distance = pt - self.center
        return torch.all(torch.abs(distance) <= self.halfwidth).item()

    def scale(self, factor: float) -> None:
        """
        Scale the AABB by a factor.
        """
        self._halfwidth *= factor

    def split(self) -> list[AABB]:
        """
        Split the AABB into 2^space_dim AABBs.

        Returns
        -------
        list[AABB]
            list of AABBs. The order of the AABBs is Morton-order.
        """
        device = self.halfwidth.device
        quarter = self.halfwidth * 0.5
        children = []

        for signs in itertools.product([-1, 1], repeat=3):
            offset = torch.tensor(signs[::-1], device=device) * quarter
            child_center = self.center + offset
            children.append(AABB(child_center, quarter))
        return children

    def merge(self, other: IAABB) -> IAABB:
        """
        Merge two AABBs.

        Notes
        -----
        For higher precision,
        the calculation is performed in the relative coordinate system
        with respect to self.center.
        """
        delta = other.center - self.center
        aabb1_min = -self.halfwidth
        aabb1_max = self.halfwidth
        aabb2_min = delta - other.halfwidth
        aabb2_max = delta + other.halfwidth
        merged_min = torch.min(aabb1_min, aabb2_min)
        merged_max = torch.max(aabb1_max, aabb2_max)
        new_center = self.center + (merged_max + merged_min) * 0.5
        new_halfwidth = (merged_max - merged_min) * 0.5
        return AABB(new_center, new_halfwidth)
