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

    Attributes
    ----------
    grid : IGridBase
        Computational grid that owns this field.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role of the field.
    num_components : int
        Number of field components ``k``.
    dimension : dict[str, int] or None
        Optional physical-dimension map.
    export : bool
        Whether the field is written to output.
    """

    @property
    @abstractmethod
    def grid(self) -> IGridBase:
        """Computational grid that owns this field."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Field name, optionally with a phase suffix."""
        pass

    @property
    @abstractmethod
    def role(self) -> FieldRole:
        """Temporal role of the field."""
        pass

    @property
    @abstractmethod
    def num_components(self) -> int:
        """Number of field components ``k``."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> dict[str, int] | None:
        """Optional physical-dimension map."""
        pass

    @property
    @abstractmethod
    def export(self) -> bool:
        """Whether the field is written to output."""
        pass


class CellField(GeometricField):
    """
    Cell-centered field class.

    Attributes
    ----------
    grid : IGridBase
        Computational grid that owns this field.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role of the field.
    num_components : int
        Number of field components ``k``.
    dimension : dict[str, int] or None
        Optional physical-dimension map from configuration.
    export : bool
        Whether the field is written to output.
    bcs : dict[PatchName, BoundaryCondition]
        Boundary conditions keyed by patch name.
    data : torch.Tensor
        Cell values with shape ``[C, k]``.
    old_data : torch.Tensor
        Previous time-level values with shape ``[C, k]``.
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

    def sync_to_grid_topology(self, *, topology_changed: bool = False) -> None:
        """
        Reallocate cell buffers when the grid cell count changes.

        Parameters
        ----------
        topology_changed : bool, default False
            If True, buffers are reallocated even when ``C`` is unchanged
            and previous values are discarded.
        """
        n_cells = self.grid.num_cells
        if (not topology_changed) and self._data.shape[0] == n_cells:
            return
        self._data = torch.zeros(
            (n_cells, self.num_components),
            dtype=self.grid.dtype,
            device=self.grid.device,
        )
        self.update_history()

    def reset_data(self, init_value: list[float]):
        """
        Reset data to initial condition.
        """
        if len(init_value) != self.num_components:
            raise ValueError(
                f"Field {self.name!r}: internal has "
                f"{len(init_value)} value(s), "
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
        """Computational grid that owns this field."""
        return self._grid

    # ================================
    # Metadata Accessors
    # ================================
    @property
    def name(self) -> str:
        """Field name, optionally with a phase suffix."""
        return self._name

    @property
    def role(self) -> FieldRole:
        """Temporal role of the field."""
        return self._role

    @property
    def num_components(self) -> int:
        """Number of field components ``k``."""
        return self._num_components

    @property
    def dimension(self) -> dict[str, int] | None:
        """Optional physical-dimension map from configuration."""
        return self._dimension

    @property
    def export(self) -> bool:
        """Whether the field is written to output."""
        return self._export

    @export.setter
    def export(self, value: bool):
        self._export = value

    @property
    def bcs(self) -> dict[PatchName, BoundaryCondition]:
        """Boundary conditions keyed by patch name."""
        return self._bcs

    # ================================
    # Data Accessors
    # ================================
    @property
    def data(self) -> Float[torch.Tensor, "C k"]:
        """Cell values with shape ``[C, k]``."""
        return self._data

    @data.setter
    def data(self, value: Float[torch.Tensor, "C k"]):
        self._data = value

    @property
    def old_data(self) -> Float[torch.Tensor, "C k"]:
        """Previous time-level values with shape ``[C, k]``."""
        return self._old_data


class FaceField(GeometricField):
    """
    Face-centered field class.

    Attributes
    ----------
    grid : IGridBase
        Computational grid that owns this field.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role of the field.
    num_components : int
        Number of field components ``k``.
    dimension : dict[str, int] or None
        Optional physical-dimension map.
    export : bool
        Whether the field is written to output.
    num_single_sided : int
        Number of ordinary single-sided internal faces.
    single_mask : torch.Tensor
        Boolean mask of single-sided internal faces, shape ``[F_internal]``.
    single_data : torch.Tensor
        Values on single-sided internal faces, shape ``[F_single, k]``.
    immersed_upper : torch.Tensor
        Upper-side immersed-face values, shape ``[F_immersed, k]``.
    immersed_lower : torch.Tensor
        Lower-side immersed-face values, shape ``[F_immersed, k]``.
    domain_bnd_data : torch.Tensor
        Domain-boundary face values, shape ``[F_bnd, k]``.
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

    def sync_to_grid_topology(self, *, topology_changed: bool = False) -> None:
        """
        Resize face buffers to match the current grid IBM / topology.

        Parameters
        ----------
        topology_changed : bool, default False
            If True, internal-face and domain-boundary counts may have
            changed and all buffers are reallocated. If False, topology is
            assumed fixed and single-sided values are preserved on faces
            that remain non-immersed.
        """
        grid = self.grid
        n_internal = grid.num_internal_faces
        n_bnd = grid.num_domain_bnd_faces
        k = self.num_components
        device = grid.device
        dtype = grid.dtype

        new_mask = torch.ones(n_internal, dtype=torch.bool, device=device)
        if isinstance(grid, AxisProjectedGrid):
            new_mask[grid.ap_is_immersed_faces] = False
            immersed_shape = (grid.num_immersed_faces, k)
            self._immersed_upper = torch.zeros(
                immersed_shape, dtype=dtype, device=device
            )
            self._immersed_lower = torch.zeros(
                immersed_shape, dtype=dtype, device=device
            )

        if topology_changed or self._single_mask.shape[0] != n_internal:
            self._single_mask = new_mask
            self._num_single_sided = int(new_mask.sum().item())
            self._single_data = torch.zeros(
                (self._num_single_sided, k), dtype=dtype, device=device
            )
            self._domain_bnd_data = torch.zeros(
                (n_bnd, k), dtype=dtype, device=device
            )
        else:
            full = torch.zeros((n_internal, k), dtype=dtype, device=device)
            full[self._single_mask] = self._single_data
            self._single_mask = new_mask
            self._single_data = full[new_mask]
            self._num_single_sided = int(new_mask.sum().item())

    # ================================
    # Grid Accessors
    # ================================
    @property
    def grid(self) -> IGridBase:
        """Computational grid that owns this field."""
        return self._grid

    # ================================
    # Metadata Accessors
    # ================================
    @property
    def name(self) -> str:
        """Field name, optionally with a phase suffix."""
        return self._name

    @property
    def role(self) -> FieldRole:
        """Temporal role of the field."""
        return self._role

    @property
    def num_components(self) -> int:
        """Number of field components ``k``."""
        return self._num_components

    @property
    def dimension(self) -> dict[str, int] | None:
        """Optional physical-dimension map."""
        return self._dimension

    @property
    def export(self) -> bool:
        """Whether the field is written to output."""
        return self._export

    # ================================
    # Data Accessors
    # ================================
    @property
    def num_single_sided(self) -> int:
        """Number of ordinary single-sided internal faces."""
        return self._num_single_sided

    @property
    def single_mask(self) -> Bool[torch.Tensor, " F_internal "]:
        """Boolean mask of single-sided internal faces ``[F_internal]``."""
        return self._single_mask

    @property
    def single_data(self) -> Float[torch.Tensor, " F_single k"]:
        """Values on single-sided internal faces ``[F_single, k]``."""
        return self._single_data

    @single_data.setter
    def single_data(self, value: Float[torch.Tensor, "F_single k"]):
        self._single_data = value

    @property
    def immersed_upper(self) -> Float[torch.Tensor, "F_immersed k"]:
        """Upper-side immersed-face values, shape ``[F_immersed, k]``."""
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
        """Lower-side immersed-face values, shape ``[F_immersed, k]``."""
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
        """Domain-boundary face values, shape ``[F_bnd, k]``."""
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
    """
    Return an existing cell field or create and register a new one.

    Parameters
    ----------
    grid : IGridBase
        Computational grid that owns the field registry.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role required for the field.
    num_components : int
        Number of components ``k``.

    Returns
    -------
    CellField
        Existing or newly created cell-centered field.

    Raises
    ------
    AssertionError
        If an existing field has a mismatched ``role`` or ``num_components``.
    """
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
    """
    Return an existing face field or create and register a new one.

    Parameters
    ----------
    grid : IGridBase
        Computational grid that owns the field registry.
    name : str
        Field name, optionally with a phase suffix.
    role : FieldRole
        Temporal role required for the field.
    num_components : int
        Number of components ``k``.
    dimension : dict[str, int] or None, optional
        Optional physical-dimension map.
    export : bool, optional
        Whether the field is written to output. Default is ``True``.

    Returns
    -------
    FaceField
        Existing or newly created face-centered field.

    Raises
    ------
    AssertionError
        If an existing field has mismatched metadata.
    """
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
