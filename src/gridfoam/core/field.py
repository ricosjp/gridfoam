from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from jaxtyping import Bool, Float

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.factory import create_boundary_condition
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
    ):
        self._grid = grid
        self._name = name
        self._role = role
        self._num_components = num_components

        device = grid.device
        dtype = grid.dtype

        shape = (grid.num_cells, num_components)
        self._data = torch.zeros(shape, dtype=dtype, device=device)

        self._bcs: dict[PatchName, BoundaryCondition] = {}
        self._dimension = None
        self._export = False
        self._condition = self.grid.sim_config.get_field_condition(self.name)
        if self._condition is not None:
            self._dimension = self._condition.dimension
            self._export = self._condition.export
            # init internal value
            self.reset_data(self._condition.internal)
            # init boundary conditions
            for bc_config in self._condition.iter_bc_configs():
                bc = create_boundary_condition(
                    bc_config, dtype=grid.dtype, device=grid.device
                )
                for patch in bc_config.patches:
                    self._bcs[patch] = bc

        # Self-register in the grid field registry.
        grid.register_cellfield(self)

        self.update_history()

    def reset_data(self, init_value: list[float]):
        """
        Reset data to initial condition.
        """
        if len(init_value) != self.num_components:
            raise ValueError(
                f"Field {self.name!r}: internal has {len(init_value)} value(s), "
                f"expected {self.num_components}"
            )
        self._data[:] = torch.tensor(
            init_value, dtype=self.grid.dtype, device=self.grid.device
        ).view(1, self.num_components)

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

    @export.setter
    def export(self, value: bool):
        self._export = value

    @property
    def bcs(self) -> dict[PatchName, BoundaryCondition]:
        return self._bcs

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
        grid.register_facefield(self)

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


def get_or_create_cellfield(
    grid: IGridBase,
    name: str,
    role: FieldRole,
    num_components: int,
) -> CellField:
    field = grid.get_cellfield(name)
    if field is None:
        field = CellField(
            grid=grid,
            name=name,
            role=role,
            num_components=num_components,
        )
    assert field.role == role
    assert field.num_components == num_components
    return field


def get_or_create_facefield(
    grid: IGridBase,
    name: str,
    role: FieldRole,
    num_components: int,
    dimension: dict[str, int] | None = None,
    export: bool = True,
) -> FaceField:
    field = grid.get_facefield(name)
    if field is None:
        field = FaceField(
            grid=grid,
            name=name,
            role=role,
            num_components=num_components,
            dimension=dimension,
            export=export,
        )
    assert field.role == role
    assert field.num_components == num_components
    assert field.dimension == dimension
    assert field.export == export
    return field
