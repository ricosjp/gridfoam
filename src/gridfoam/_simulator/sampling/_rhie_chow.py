from gridfoam._base._tensor_grid import TensorGrid
from gridfoam.cubion import NodeType


class RhieChowInterpolation:
    def __init__(self, velocity_name: str):
        self.U_i_name = velocity_name
        self.U_f_name = "_" + velocity_name + "_f"

    def run(self, grid: TensorGrid) -> None:
        for octree_level in grid.data.octree_levels:
            for cube in octree_level.nodes.values():
                if cube.node_type != NodeType.LEAF:
                    continue
                U_i = cube.old.cells[self.U_i_name]
                U_f = U_i.face_average()  # TODO: Add Rhie-Chow correction
                cube.old.faces[self.U_f_name] = U_f
