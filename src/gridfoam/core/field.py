from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from jaxtyping import Bool, Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.enums import FieldRole
from gridfoam.meta.types import PatchName


class GeometricField(ABC):
    """
    Abstract base class for geometric fields on a mesh.

    Holds common metadata such as name, grid, component count, and role.
    """

    @property
    @abstractmethod
    def grid(self) -> IGridBase:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def role(self) -> FieldRole:
        pass

    @property
    @abstractmethod
    def num_components(self) -> int:
        pass

    @property
    @abstractmethod
    def dimension(self) -> dict[str, int] | None:
        pass

    @property
    @abstractmethod
    def export(self) -> bool:
        pass


class CellField(GeometricField):
    """
    Cell-centered field class.
    """

    def __init__(
        self,
        grid: IGridBase,
        name: str,
        role: FieldRole,
        num_components: int,
        dimension: dict[str, int] | None = None,
        export: bool = True,
        bcs: dict[PatchName, BoundaryCondition] | None = None,
        ref_cell_id: int = 0,
        ref_value: float = 0.0,
    ):
        self._grid = grid
        self._name = name
        self._role = role
        self._num_components = num_components
        self._dimension = dimension
        self._export = export
        self._bcs = bcs or {}
        self._ref_cell_id = ref_cell_id
        self._ref_value = ref_value

        device = grid.device
        dtype = grid.dtype

        shape = (grid.num_cells, num_components)
        self._data = torch.zeros(shape, dtype=dtype, device=device)
        self.update_history()

        # Self-register in the grid field registry.
        grid.register_field(self)

    def update_history(self):
        """
        Update time history at the end of a time step.
        """
        if self._role == FieldRole.TRANSIENT:
            self._old_data = self._data.clone()
            return
        self._old_data = self._data

    def add_boundary_conditions(self, bcs: dict[PatchName, BoundaryCondition]):
        """
        Add or update boundary conditions.
        """
        self._bcs.update(bcs)

    # ================================
    # Grid Accessors
    # ================================
    @property
    def grid(self) -> IGridBase:
        return self._grid

    # ================================
    # Metadata Accessors
    # ================================
    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> FieldRole:
        return self._role

    @property
    def num_components(self) -> int:
        return self._num_components

    @property
    def dimension(self) -> dict[str, int] | None:
        return self._dimension

    @property
    def export(self) -> bool:
        return self._export

    @property
    def bcs(self) -> dict[PatchName, BoundaryCondition]:
        return self._bcs

    @property
    def ref_cell_id(self) -> int:
        return self._ref_cell_id

    @property
    def ref_value(self) -> float:
        return self._ref_value

    # ================================
    # Data Accessors
    # ================================
    @property
    def data(self) -> Float[torch.Tensor, "C k"]:
        return self._data

    @data.setter
    def data(self, value: Float[torch.Tensor, "C k"]):
        self._data = value

    @property
    def old_data(self) -> Float[torch.Tensor, "C k"]:
        return self._old_data


class FaceField(GeometricField):
    """
    Face-centered field class.
    """

    def __init__(
        self,
        grid: IGridBase,
        name: str,
        role: FieldRole,
        num_components: int,
        dimension: dict[str, int] | None = None,
        export: bool = True,
    ):
        self._grid = grid
        self._name = name
        self._role = role
        self._num_components = num_components
        self._dimension = dimension
        self._export = export

        device = grid.device
        dtype = grid.dtype

        # Domain-boundary data
        domain_bnd_shape = (grid.num_domain_bnd_faces, num_components)
        self._domain_bnd_data = torch.zeros(
            domain_bnd_shape, dtype=dtype, device=device
        )

        self._num_single_sided = grid.num_internal_faces
        self._single_mask = torch.ones(
            grid.num_internal_faces, dtype=torch.bool, device=device
        )
        # Immersed-boundary data (double-sided)
        if isinstance(grid, AxisProjectedGrid):
            self._single_mask[grid.ap_is_immersed_faces] = False
            immersed_bnd_shape = (grid.num_immersed_faces, num_components)
            self._immersed_upper = torch.zeros(
                immersed_bnd_shape, dtype=dtype, device=device
            )
            self._immersed_lower = torch.zeros(
                immersed_bnd_shape, dtype=dtype, device=device
            )
            self._num_single_sided -= grid.num_immersed_faces

        # Internal-face data (single-sided subset only)
        shape = (self._num_single_sided, num_components)
        self._single_data = torch.zeros(shape, dtype=dtype, device=device)

        # Self-register in the grid field registry.
        grid.register_field(self)

    # ================================
    # Grid Accessors
    # ================================
    @property
    def grid(self) -> IGridBase:
        return self._grid

    # ================================
    # Metadata Accessors
    # ================================
    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> FieldRole:
        return self._role

    @property
    def num_components(self) -> int:
        return self._num_components

    @property
    def dimension(self) -> dict[str, int] | None:
        return self._dimension

    @property
    def export(self) -> bool:
        return self._export

    # ================================
    # Data Accessors
    # ================================
    @property
    def num_single_sided(self) -> int:
        return self._num_single_sided

    @property
    def single_mask(self) -> Bool[torch.Tensor, " F_internal "]:
        return self._single_mask

    @property
    def single_data(self) -> Float[torch.Tensor, " F_single k"]:
        return self._single_data

    @single_data.setter
    def single_data(self, value: Float[torch.Tensor, "F_single k"]):
        self._single_data = value

    @property
    def immersed_upper(self) -> Float[torch.Tensor, "F_immersed k"]:
        if not isinstance(self.grid, AxisProjectedGrid):
            raise ValueError(
                "Immersed upper data is not available for this grid."
            )
        return self._immersed_upper

    @immersed_upper.setter
    def immersed_upper(self, value: Float[torch.Tensor, "F_immersed k"]):
        if not isinstance(self.grid, AxisProjectedGrid):
            raise ValueError(
                "Immersed upper data is not available for this grid."
            )
        self._immersed_upper = value

    @property
    def immersed_lower(self) -> Float[torch.Tensor, "F_immersed k"]:
        if not isinstance(self.grid, AxisProjectedGrid):
            raise ValueError(
                "Immersed lower data is not available for this grid."
            )
        return self._immersed_lower

    @immersed_lower.setter
    def immersed_lower(self, value: Float[torch.Tensor, "F_immersed k"]):
        if not isinstance(self.grid, AxisProjectedGrid):
            raise ValueError(
                "Immersed lower data is not available for this grid."
            )
        self._immersed_lower = value

    @property
    def domain_bnd_data(self) -> Float[torch.Tensor, "F_bnd k"]:
        return self._domain_bnd_data

    @domain_bnd_data.setter
    def domain_bnd_data(self, value: Float[torch.Tensor, "F_bnd k"]):
        self._domain_bnd_data = value
