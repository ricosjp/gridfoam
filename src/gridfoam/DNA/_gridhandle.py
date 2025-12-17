import abc
import pathlib
from collections.abc import Iterator

import pyvista as pv
import torch
from jaxtyping import Float

from gridfoam.DNA._grid._grid import PyGrid as Grid
from gridfoam.DNA._grid._grid import PyOctreeLevel, PyOctreeNode
from gridfoam.DNA.config import GridfoamConfig
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta


class IGridHandle(abc.ABC):
    @abc.abstractmethod
    def iter_levels(self) -> Iterator[PyOctreeLevel]:
        pass

    @abc.abstractmethod
    def iter_leaf_on_level(
        self, level: PyOctreeLevel
    ) -> Iterator[PyOctreeNode]:
        pass

    @abc.abstractmethod
    def iter_gfp_on_level(self, level: PyOctreeLevel) -> Iterator[PyOctreeNode]:
        pass

    @abc.abstractmethod
    def iter_gfc_on_level(self, level: PyOctreeLevel) -> Iterator[PyOctreeNode]:
        pass

    @abc.abstractmethod
    def iter_all_leaves(self) -> Iterator[tuple[int, PyOctreeNode]]:
        pass

    @abc.abstractmethod
    def allocate_field(self, field_meta: FieldMeta) -> None:
        pass

    @abc.abstractmethod
    def allocate_equation(self, equation_meta: EquationMeta) -> None:
        pass

    @abc.abstractmethod
    def update_fvmatrix(self, equation_meta: EquationMeta) -> None:
        pass

    ## Synchronizing halo
    @abc.abstractmethod
    def sync_halo_at_depth(self, depth: int, fm_list: list[FieldMeta]) -> None:
        """
        Synchronize the halo for a given depth.

        Parameters
        ----------
        depth : int
            Depth to synchronize the halo at.
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        """
        pass

    @abc.abstractmethod
    def sync_halo(self, fm_list: list[FieldMeta]) -> None:
        """
        Synchronize the halo for all depths.

        Parameters
        ----------
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        """
        pass

    ## Synchronizing ghost from parent
    @abc.abstractmethod
    def sync_gfp_at_depth(self, depth: int, fm_list: list[FieldMeta]) -> None:
        """
        Synchronize ghost cubes from their parent cubes at a given depth.

        This method extracts data from parent cubes and distributes it to
        ghost cubes that are refined versions of their parent cubes.
        The data is interpolated using trilinear interpolation.

        Parameters
        ----------
        depth : int
            The depth level to synchronize.
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        """
        pass

    @abc.abstractmethod
    def sync_gfp(self, fm_list: list[FieldMeta]) -> None:
        """
        Synchronize ghost cubes from their parent cubes for all depths.

        This method extracts data from parent cubes and distributes it to
        ghost cubes that are refined versions of their parent cubes.
        The data is interpolated using trilinear interpolation.

        Parameters
        ----------
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        """
        pass

    ## Synchronizing ghost from child
    @abc.abstractmethod
    def sync_gfc_at_depth(self, depth: int, fm_list: list[FieldMeta]) -> None:
        """
        Synchronize ghost cubes from their child cubes at a given depth.

        This method aggregates data from child cubes and distributes it to
        ghost cubes that are coarsened versions of their child cubes.
        The data is coarsened using average pooling
        (2x2x2 cells are averaged into 1 cell).

        Parameters
        ----------
        depth : int
            The depth level to synchronize.
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        """
        pass

    @abc.abstractmethod
    def sync_gfc(self, fm_list: list[FieldMeta]) -> None:
        """
        Synchronize ghost cubes from their child cubes for all depths.

        This method aggregates data from child cubes
        and distributes it to ghost cubes
        that are coarsened versions of their child cubes.
        The data is coarsened using average pooling
        (2x2x2 cells are averaged into 1 cell).
        """
        pass

    @abc.abstractmethod
    def sync_all(self, fm_list: list[FieldMeta]) -> None:
        """
        Synchronize all fields for all depths.
        """
        pass

    @abc.abstractmethod
    def get_dx_at_depth(self, depth: int) -> Float[torch.Tensor, " 3"]:
        pass

    @abc.abstractmethod
    def save(self, path: pathlib.Path) -> None:
        pass

    @property
    @abc.abstractmethod
    def grid(self) -> Grid:
        pass

    @property
    @abc.abstractmethod
    def mesh(self) -> pv.PolyData:
        pass

    @property
    @abc.abstractmethod
    def config(self) -> GridfoamConfig:
        pass
