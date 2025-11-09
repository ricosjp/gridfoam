import numpy as np
import torch

from gridfoam._base._equation._api import ddt, div, eq_zero
from gridfoam._base._field._grid import PyGrid as Grid
from gridfoam._base._field._handle import FieldHandle
from gridfoam._base._field._registry import FieldRegistry
from gridfoam._base._schemes._registry import SchemeRegistry
from gridfoam._base._schemes._rhie_chow import RhieChowScheme
from gridfoam._base._solver._registry import SolverRegistry
from gridfoam.config import GridfoamConfig
from gridfoam.utils.enums import FieldLayout, FieldRole
from gridfoam.utils.grid_index import generate_grid_indices


class TransientSystem:
    def __init__(self, gridfoam_config: GridfoamConfig, grid: Grid):
        self.field_registry = FieldRegistry()
        T = self.field_registry.declare(
            name="temperature",
            channels=1,
            dtype=torch.float32,
            device=gridfoam_config.device,
            role=FieldRole.STATE,
            layout=FieldLayout.CELL,
            alias="T",
        )
        U = self.field_registry.declare(
            name="velocity",
            channels=3,
            dtype=torch.float32,
            device=gridfoam_config.device,
            role=FieldRole.STATE,
            layout=FieldLayout.CELL,
            alias="U",
        )
        nu = self.field_registry.declare(
            name="kinetic_viscosity",
            channels=1,
            dtype=torch.float32,
            device=gridfoam_config.device,
            role=FieldRole.AUXILIARY,
            layout=FieldLayout.CELL,
            alias="nu",
        )
        p = self.field_registry.declare(
            name="pressure",
            channels=1,
            dtype=torch.float32,
            device=gridfoam_config.device,
            role=FieldRole.STATE,
            layout=FieldLayout.CELL,
            alias="p",
        )

        # equation = eq_zero(ddt(T) + div(U, T) - laplacian(nu, T), tag="heat")
        equation = eq_zero(ddt(T) + div(U, T), tag="heat")
        self.target = T

        self.rhie_chow = RhieChowScheme(U, p, self.field_registry)
        self.scheme_registry = SchemeRegistry(
            gridfoam_config.simulator.fvSchemes
        )
        self.solver_registry = SolverRegistry(
            gridfoam_config.simulator.fvSolution, self.field_registry
        )

        self.fvm_system = self.scheme_registry.resolve(equation.expr)
        self.solver = self.solver_registry.resolve(equation)

        self.finest_depth = grid.max_depth - 1
        self._deltaT = gridfoam_config.simulator.control.deltaT
        self._time = 0.0
        self._step = 0
        # allocate fields to the storage
        FieldHandle.allocate(gridfoam_config.cube, self.field_registry, grid)

    def setup(self) -> None:
        T = self.target
        U = self.field_registry.get_field_desc_by_alias("U")
        nu = self.field_registry.get_field_desc_by_alias("nu")
        for octree_level in FieldHandle.iter_octree_levels():
            depth = octree_level.depth
            half_dx = 0.5 * FieldHandle.get_dx_at(depth)
            for cube in FieldHandle.iter_leaf_cubes_of(octree_level):
                divisions = torch.tensor(
                    [FieldHandle._N] * 3, dtype=torch.int64
                )
                global_index = torch.tensor(
                    cube.cubecode.to_global_index(depth).astype(np.int64),
                    dtype=torch.int64,
                )
                global_lower_position = torch.tensor(
                    FieldHandle._grid.domain.lower, dtype=torch.float32
                )
                local_cell_indices = generate_grid_indices(divisions).reshape(
                    3, FieldHandle._N, FieldHandle._N, FieldHandle._N
                )
                global_cell_indices = (
                    global_index[:, None, None, None] * FieldHandle._N
                    + local_cell_indices
                )
                global_cell_positions = (
                    global_lower_position[:, None, None, None]
                    + (2 * global_cell_indices + 1)
                    * half_dx[:, None, None, None]
                )
                x = global_cell_positions[0]
                # y = global_cell_positions[1]
                z = global_cell_positions[2]

                # Set U to (1,0,0)
                # cube.old.cells[U].interior[0] = 1.0
                cube.old.cells[U.canonical_name].interior[0] = 0.707
                cube.old.cells[U.canonical_name].interior[2] = 0.707

                # Set T to 1.0 where -2 < x < 0 and 1 < z < 3
                mask = (-2.0 < x) & (x < 0.0) & (1.0 < z) & (z < 3.0)
                cube.old.cells[T.canonical_name].interior[0, mask] = 1.0

                # Set nu to 0.01
                cube.old.cells[nu.canonical_name].interior = torch.full(
                    (1, FieldHandle._N, FieldHandle._N, FieldHandle._N),
                    0.01,
                )
        FieldHandle.sync_all([T, U, nu])

    def advance(self) -> None:
        """
        Advance the solution by one time step.
        """
        for octree_level in FieldHandle.iter_octree_levels():
            depth = octree_level.depth
            dx = FieldHandle.get_dx_at(depth)
            dt = self._deltaT / (1 << depth)
            for cube in FieldHandle.iter_leaf_cubes_of(octree_level):
                self.rhie_chow.interpolate(cube, dt, dx)
        dt = self._deltaT / (1 << self.finest_depth)
        # dt = self._deltaT
        self.solver.solve(self.target, self.fvm_system, dt, dx)
        self._time += self._deltaT
        self._step += 1

    @property
    def time(self) -> float:
        return self._time

    @property
    def step(self) -> int:
        return self._step
