import numpy as np
import torch

from gridfoam._base import TensorGrid, iter_leaf_cubes_of
from gridfoam._io import save_grid
from gridfoam._simulator._scheme import Ddt, Div, RhieChowInterpolation
from gridfoam._simulator._solver import BiCGSTAB
from gridfoam.cubion import PyCubeCode
from gridfoam.utils.grid_index import generate_grid_indices


class TransientSolver:
    """
    Transient solver for finite volume method with adaptive mesh refinement.

    This class implements a transient solver that advances the solution
    in time using a hierarchical time-stepping scheme with adaptive
    mesh refinement (AMR). It handles the coupling between different
    refinement levels and manages the time integration process.
    """

    def __init__(self, grid: TensorGrid):
        """
        Initialize the transient solver.

        Parameters
        ----------
        grid : TensorGrid
            The tensor grid containing the simulation data.
        """
        self.grid = grid
        self.deltaT = grid.config.simulator.control.deltaT
        self.end_time = grid.config.simulator.control.endTime
        self.write_interval = grid.config.simulator.control.writeInterval
        self.finest_depth = grid.data.max_depth - 1
        self.w_interior = grid.config.cube.interior_width
        self.rhie_chow = RhieChowInterpolation(velocity_name="U")
        self.solver = BiCGSTAB(Ddt("T") + Div("U", "T"))
        self.solver.configure(max_iter=1000, tol=1e-6)

    def _solve_level(self, depth: int) -> None:
        """
        Solve the solution at a specific depth level.
        """
        octree_level = self.grid.data.octree_levels[depth]
        bounds = torch.tensor(octree_level.bounds * self.w_interior)
        dx = self.grid.domain_width / bounds
        T_new = self.solver.solve(octree_level, self.deltaT, dx)
        if torch.any(torch.isnan(T_new)):
            raise ValueError("T_new is nan")
        n_cells_per_cube = octree_level.n_cells_per_node
        for i, cube in enumerate(iter_leaf_cubes_of(octree_level)):
            _slice = slice(i * n_cells_per_cube, (i + 1) * n_cells_per_cube)
            shape = cube.cur.cells["T"].interior_shape
            cube.cur.cells["T"].interior = T_new[_slice].reshape(shape)
            cube.cur.cells["T"], cube.old.cells["T"] = (
                cube.old.cells["T"],
                cube.cur.cells["T"],
            )
        return

    def _advance_level_step(self, depth: int) -> None:
        """
        Advance the solution by one time step at a specific depth level.

        This method implements the hierarchical time-stepping scheme where
        finer levels are advanced multiple times for each coarse level step.
        The finest level is solved directly, while coarser levels coordinate
        with their children through ghost cell synchronization.

        Parameters
        ----------
        depth : int
            The depth level to advance.
        """
        if depth == self.finest_depth:
            self._solve_level(depth)
            self.grid.update_halo_at_depth(depth)
            return
        for _ in range(2):
            self.grid.sync_ghost_from_parent_at_depth(depth + 1)
            self.grid.update_halo_at_depth(depth + 1)
            self._advance_level_step(depth + 1)
            self.grid.sync_ghost_from_children_at_depth(depth)
            self.grid.update_halo_at_depth(depth)
        self._solve_level(depth)

    def setup(self) -> None:
        """
        Set up the initial conditions for the simulation.

        This method initializes the velocity and temperature fields,
        allocates tensor memory, and sets up the initial conditions.
        The velocity is set to (1,0,0) and temperature is set to 1.0
        in the region where 0 < z < 4.
        """
        self.grid.add_cell_field("U", (3,), torch.float32)
        self.grid.add_cell_field("T", (1,), torch.float32)
        self.grid.allocate_field_tensors()
        for octree_level in self.grid.data.octree_levels:
            depth = octree_level.depth
            bounds = torch.tensor(octree_level.bounds * self.w_interior)
            half_dx = 0.5 * self.grid.domain_width / bounds
            for cube in octree_level.nodes.values():
                cubecode: PyCubeCode = cube.cubecode
                divisions = torch.tensor(
                    [self.w_interior] * 3, dtype=torch.int64
                )
                global_index = torch.tensor(
                    cubecode.to_global_index(depth).astype(np.int64),
                    dtype=torch.int64,
                )
                global_lower_position = torch.tensor(
                    self.grid.data.domain.lower, dtype=torch.float32
                )
                local_cell_indices = generate_grid_indices(divisions).reshape(
                    3, self.w_interior, self.w_interior, self.w_interior
                )
                global_cell_indices = (
                    global_index[:, None, None, None] * self.w_interior
                    + local_cell_indices
                )
                global_cell_positions = (
                    global_lower_position[:, None, None, None]
                    + (2 * global_cell_indices + 1)
                    * half_dx[:, None, None, None]
                )

                x = global_cell_positions[0]
                y = global_cell_positions[1]
                z = global_cell_positions[2]

                # Set U to (1,0,0)
                cube.old.cells["U"].interior = torch.zeros(
                    3, self.w_interior, self.w_interior, self.w_interior
                )
                cube.old.cells["U"].interior[0] = 1.0
                # cube.old.cells["U"].interior[0] = 0.707
                # cube.old.cells["U"].interior[2] = 0.707
                # Set T to 1.0 where -2 < x < 0 and 1 < z < 3
                mask = (-2.0 < x) & (x < 0.0) & (1.0 < z) & (z < 3.0)
                cube.old.cells["T"].interior[0, mask] = 1.0

        self.grid.update_halo()

    def solve(self) -> None:
        """
        Run the transient simulation.

        This method advances the solution in time until the end time
        is reached. It performs Rhie-Chow interpolation, advances
        the solution using hierarchical time-stepping, and saves
        output at specified intervals.
        """
        step = 0
        time = 0.0
        deltaT = self.deltaT
        save_grid(self.grid, f"grid_{step:04d}.vtkhdf")
        while 1:
            print(f"start step: {step}")
            self.rhie_chow.run(self.grid)
            self._advance_level_step(0)
            time += deltaT
            step += 1
            if step % self.write_interval == 0:
                save_grid(self.grid, f"grid_{step:04d}.vtkhdf")
            if time >= self.end_time:
                break
