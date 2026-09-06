from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import torch
from jaxtyping import Bool, Float
from phlower_tensor import PhysicalDimensions

from gridfoam.boundaries.base import BoundaryCondition
from gridfoam.boundaries.factory import create_boundary_condition
from gridfoam.core.dimensions import (
    DimensionLike,
    assert_compatible,
    resolve_field_dimension,
    to_dimensions,
)
from gridfoam.core.fv_cache import FvFieldCache
from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.enums import FieldRole
from gridfoam.meta.types import PatchName


def _tensor_token(tensor: torch.Tensor) -> tuple[int, int, int]:
    """Identity, storage pointer and in-place version of ``tensor``."""
    return (id(tensor), tensor.data_ptr(), tensor._version)


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
    dimension : PhysicalDimensions or None
        Optional physical dimension of the field.
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
    def dimension(self) -> PhysicalDimensions | None:
        """Optional physical dimension of the field."""
        pass

    @property
    @abstractmethod
    def export(self) -> bool:
        """Whether the field is written to output."""
        pass

    @abstractmethod
    def state_token(self) -> tuple[int, ...]:
        """
        Fingerprint of the current data used for cache invalidation.

        Returns
        -------
        tuple[int, ...]
            Changes whenever the field data is replaced or modified in place.
        """
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
    dimension : PhysicalDimensions or None
        Optional physical dimension of the field.
    export : bool
        Whether the field is written to output.
    bcs : dict[PatchName, BoundaryCondition]
        Boundary conditions keyed by patch name.
    data : torch.Tensor
        Cell values with shape ``[C, k]``.
    old_data : torch.Tensor
        Previous time-level values with shape ``[C, k]``.
    fv_cache : FvFieldCache
        Field-owned FV cache (boundary batches and states).
    """

    def __init__(
        self,
        grid: IGridBase,
        name: str,
        role: FieldRole,
        num_components: int,
        dimension: DimensionLike = None,
    ):
        self._grid = grid
        self._name = name
        self._role = role
        self._num_components = num_components
        self._fv_cache = FvFieldCache()

        device = grid.device
        dtype = grid.dtype

        shape = (grid.num_cells, num_components)
        self._data = torch.zeros(shape, dtype=dtype, device=device)

        self._bcs: dict[PatchName, BoundaryCondition] = {}
        self._export = False
        self._condition = self.grid.sim_config.get_field_condition(self.name)
        config_dimension = (
            self._condition.dimension if self._condition is not None else None
        )
        self._dimension = resolve_field_dimension(
            self.name, explicit=dimension, config=config_dimension
        )
        # A caller-supplied dimension wins over the configured one, so reject
        # the combination instead of silently discarding the configuration.
        assert_compatible(
            self._dimension,
            to_dimensions(config_dimension),
            f"cell field {name!r} configuration",
        )
        if self._condition is not None:
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
        self.fv_cache.clear()
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
        self.fv_cache.clear_boundary_batches()

    @property
    def fv_cache(self) -> FvFieldCache:
        """Field-owned FV cache (boundary batches and states)."""
        return self._fv_cache

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
    def dimension(self) -> PhysicalDimensions | None:
        """Optional physical dimension of the field."""
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

    def state_token(self) -> tuple[int, ...]:
        """
        Fingerprint of the current cell data used for cache invalidation.

        Combines the tensor identity, storage pointer and in-place version
        counter, so both ``field.data = new`` and ``field.data[:] = x``
        change the token.
        """
        return _tensor_token(self._data)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """
        Move cell buffers to ``device`` in place.

        Parameters
        ----------
        device : torch.device or str
            Target device.
        non_blocking : bool, default False
            Passed through to ``Tensor.to``.

        Returns
        -------
        CellField
            This field after the buffers have been moved.
        """
        old_aliased = self._old_data is self._data
        self._data = self._data.to(device=device, non_blocking=non_blocking)
        if old_aliased:
            self._old_data = self._data
        else:
            self._old_data = self._old_data.to(
                device=device, non_blocking=non_blocking
            )
        return self


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
    dimension : PhysicalDimensions or None
        Optional physical dimension of the field.
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

    Notes
    -----
    Packed layout along axis 0 is
    ``[single | domain_bnd | immersed_upper | immersed_lower]``.
    Immersed blocks are present only on ``AxisProjectedGrid``.
    """

    def __init__(
        self,
        grid: IGridBase,
        name: str,
        role: FieldRole,
        num_components: int,
        dimension: DimensionLike = None,
        export: bool = True,
    ):
        self._grid = grid
        self._name = name
        self._role = role
        self._num_components = num_components
        self._dimension = resolve_field_dimension(name, explicit=dimension)
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

        self.update_history()

    def update_history(self) -> None:
        """
        Store the current single-sided face values as the old time level.

        An independent copy is always kept so that ``fvc.ddt_corr`` sees the
        previous time-step flux even when the current values are updated in
        place. The flux field is created as ``FieldRole.LOCAL`` by several
        producers (potential-flow initialisation, diagnostics), so the role
        is not used to decide whether to copy.
        """
        self._old_single_data = self._single_data.clone()

    @property
    def old_single_data(self) -> Float[torch.Tensor, " F_single k"]:
        """Previous time-level single-sided values, ``[F_single, k]``."""
        return self._old_single_data

    def state_token(self) -> tuple[int, ...]:
        """
        Fingerprint of all face blocks used for cache invalidation.

        See :meth:`CellField.state_token`.
        """
        token: tuple[int, ...] = ()
        for block in self._pack_blocks():
            token += _tensor_token(block)
        return token

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
        # The previous time level is undefined on the new face set.
        self.update_history()

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
    def dimension(self) -> PhysicalDimensions | None:
        """Optional physical dimension of the field."""
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

    def packed_n_rows(self) -> int:
        """Packed layout row count (feature axis ``k`` is not included)."""
        return sum(block.shape[0] for block in self._pack_blocks())

    def pack(self) -> Float[torch.Tensor, "N k"]:
        """
        Concatenate face blocks along axis 0.

        Layout is ``[single | domain_bnd | immersed_upper | immersed_lower]``.
        Immersed blocks are omitted unless the grid is an
        ``AxisProjectedGrid``.
        """
        return torch.cat(self._pack_blocks(), dim=0)

    def unpack(self, packed: Float[torch.Tensor, "N k"]) -> None:
        """
        Write a packed tensor back into the face blocks.

        Parameters
        ----------
        packed : torch.Tensor
            Packed values with shape ``[N, k]``.

        Raises
        ------
        ValueError
            If the row count differs from ``packed_n_rows()`` or the feature
            axis ``k`` differs from ``num_components``.
        """
        n_rows = self.packed_n_rows()
        n_features = self.num_components
        if packed.shape[0] != n_rows:
            raise ValueError(
                f"packed has {packed.shape[0]} rows, expected {n_rows}"
            )
        if packed.shape[1] != n_features:
            raise ValueError(
                f"packed has {packed.shape[1]} features, expected {n_features}"
            )
        offset = 0
        for block in self._pack_blocks():
            n_block = block.shape[0]
            block[:] = packed[offset : offset + n_block]
            offset += n_block

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """
        Move face buffers to ``device`` in place.

        Parameters
        ----------
        device : torch.device or str
            Target device.
        non_blocking : bool, default False
            Passed through to ``Tensor.to``.

        Returns
        -------
        FaceField
            This field after the buffers have been moved.
        """
        self._single_data = self._single_data.to(
            device=device, non_blocking=non_blocking
        )
        self._old_single_data = self._old_single_data.to(
            device=device, non_blocking=non_blocking
        )
        self._domain_bnd_data = self._domain_bnd_data.to(
            device=device, non_blocking=non_blocking
        )
        self._single_mask = self._single_mask.to(
            device=device, non_blocking=non_blocking
        )
        if isinstance(self.grid, AxisProjectedGrid):
            self._immersed_upper = self._immersed_upper.to(
                device=device, non_blocking=non_blocking
            )
            self._immersed_lower = self._immersed_lower.to(
                device=device, non_blocking=non_blocking
            )
        return self

    def _pack_blocks(self) -> list[torch.Tensor]:
        """Face-value blocks in packed-layout order."""
        blocks = [self._single_data, self._domain_bnd_data]
        if isinstance(self.grid, AxisProjectedGrid):
            blocks.extend([self._immersed_upper, self._immersed_lower])
        return blocks


def packed_face_n_rows(grid: IGridBase) -> int:
    """
    Return the packed face-layout row count for ``grid``.

    Parameters
    ----------
    grid : IGridBase
        Computational grid whose topology defines the packed layout.

    Returns
    -------
    int
        Number of rows ``N`` in ``[N, k]`` packed face tensors.
    """
    if isinstance(grid, AxisProjectedGrid):
        n_single = grid.num_internal_faces - grid.num_immersed_faces
        return (
            n_single + grid.num_domain_bnd_faces + 2 * grid.num_immersed_faces
        )
    return grid.num_internal_faces + grid.num_domain_bnd_faces


def get_or_create_cellfield(
    grid: IGridBase,
    name: str,
    role: FieldRole,
    num_components: int,
    dimension: DimensionLike = None,
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
    dimension : PhysicalDimensions or dict[str, float or int] or None, optional
        Optional physical dimension of the field.

    Returns
    -------
    CellField
        Existing or newly created cell-centered field.

    Raises
    ------
    AssertionError
        If an existing field has a mismatched ``role`` or ``num_components``.
    DimensionMismatchError
        If an existing field has an incompatible dimension.
    """
    resolved = resolve_field_dimension(name, explicit=dimension)
    field = grid.get_cellfield(name)
    if field is None:
        field = CellField(
            grid=grid,
            name=name,
            role=role,
            num_components=num_components,
            dimension=dimension,
        )
    assert field.role == role
    assert field.num_components == num_components
    assert_compatible(field.dimension, resolved, f"cell field {name!r}")
    return field


def get_or_create_facefield(
    grid: IGridBase,
    name: str,
    role: FieldRole,
    num_components: int,
    dimension: DimensionLike = None,
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
    dimension : PhysicalDimensions or dict[str, float or int] or None, optional
        Optional physical dimension of the field.
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
    DimensionMismatchError
        If an existing field has an incompatible dimension.
    """
    resolved = resolve_field_dimension(name, explicit=dimension)
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
    assert_compatible(field.dimension, resolved, f"face field {name!r}")
    assert field.export == export
    return field
