import numpy as np
import torch

from gridfoam._base._tensor_grid import TensorGrid
from gridfoam._io import save_grid
from gridfoam._simulator._equation import Ddt, Div
from gridfoam._simulator._solver import BiCGSTAB
from gridfoam._simulator.sampling import RhieChowInterpolation
from gridfoam.cubion import PyCubeCode
from gridfoam.utils.grid_index import generate_grid_indices


class TransientSolver:
    def __init__(self, grid: TensorGrid):
        self.grid = grid
        self.deltaT = grid.config.simulator.control.deltaT
        self.end_time = grid.config.simulator.control.endTime
        self.write_interval = grid.config.simulator.control.writeInterval
        self.finest_depth = grid.data.max_depth - 1
        self.w_interior = grid.config.cube.interior_width
        self.rhie_chow = RhieChowInterpolation(velocity_name="U")
        self.solver = BiCGSTAB(Ddt("T") + Div("U", "T"))
        self.solver.configure(max_iter=1000, tol=1e-6)

    def _advance_level_step(self, depth: int) -> None:
        if depth == self.finest_depth:
            octree_level = self.grid.data.octree_levels[depth]
            bounds = torch.tensor(octree_level.bounds * self.w_interior)
            dx = self.grid.domain_width / bounds
            T_new = self.solver.solve(octree_level, self.deltaT, dx)
            n_cells_per_cube = octree_level.n_cells_per_node
            for i, cube in enumerate(octree_level.nodes.values()):
                _slice = slice(
                    i * n_cells_per_cube, (i + 1) * n_cells_per_cube
                )
                shape = cube.cur.cells["T"].interior_shape
                cube.cur.cells["T"].interior = T_new[_slice].reshape(shape)
                cube.cur, cube.old = cube.old, cube.cur
            return
        for _ in range(2):
            self.grid.sync_ghost_from_parent_at_depth(depth + 1)
            self._advance_level_step(depth + 1)
            self.grid.sync_ghost_from_children_at_depth(depth)
        self._advance_level_step(depth)

    def setup(self) -> None:
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
                local_cell_indices = generate_grid_indices(divisions).reshape(
                    3, self.w_interior, self.w_interior, self.w_interior
                )
                global_cell_indices = (
                    global_index[:, None, None, None] * self.w_interior
                    + local_cell_indices
                )
                global_lower_position = torch.tensor(
                    self.grid.data.domain.lower, dtype=torch.float32
                )
                global_cell_position = (
                    global_lower_position[:, None, None, None]
                    + (
                        2
                        * global_cell_indices.to(
                            dtype=global_lower_position.dtype
                        )
                        + 1
                    )
                    * half_dx[:, None, None, None]
                )
                # Set U to (1,0,0) and T to 1.0 where 0 < z < 4
                cube.old.cells["U"].interior = torch.zeros(3, self.w_interior, self.w_interior, self.w_interior)
                cube.old.cells["U"].interior[0] = 1.0
                mask = (0.0 < global_cell_position[2]) & (global_cell_position[2] < 4.0)
                cube.old.cells["T"].interior[0, mask] = 1.0

        self.grid.update_halo()
        self.rhie_chow.run(self.grid)

    def solve(self) -> None:
        step = 0
        time = 0.0
        deltaT = self.deltaT
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
