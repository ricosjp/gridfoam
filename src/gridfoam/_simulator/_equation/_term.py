import torch
from jaxtyping import Float

from gridfoam._base._face_tensor import FaceTensor
from gridfoam._simulator._equation._interface import FVMTerm
from gridfoam.cubion import NodeType, PyOctreeLevel


class Ddt(FVMTerm):
    def __init__(self, fieldname: str):
        """
        Parameters
        ----------
        fieldname : str
            Name of the field to be integrated over time.
        """
        self.fieldname = fieldname

    def __add__(self, other: FVMTerm) -> FVMTerm:
        return Expr(self, other, "+")

    def __sub__(self, other: FVMTerm) -> FVMTerm:
        return Expr(self, other, "-")

    def matvec(
        self,
        octree_level: PyOctreeLevel,
        x: torch.Tensor,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        depth = octree_level.depth
        rdt = (1 << depth) / dt
        return rdt * x

    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        depth = octree_level.depth
        rdt = (1 << depth) / dt
        return torch.full((octree_level.n_cells,), rdt)

    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        depth = octree_level.depth
        rdt = (1 << depth) / dt
        n_cells_per_cube = octree_level.n_cells_per_node
        b = torch.zeros(octree_level.n_cells)
        for i, cube in enumerate(octree_level.nodes.values()):
            if cube.node_type != NodeType.LEAF:
                continue
            _slice = slice(i * n_cells_per_cube, (i + 1) * n_cells_per_cube)
            b[_slice] = rdt * cube.old.cells[self.fieldname].interior.reshape(
                -1
            )
        return b


class Div(FVMTerm):
    def __init__(self, velocity_name: str, fieldname: str):
        self.fieldname = fieldname
        self.U_f_name = "_" + velocity_name + "_f"

    def __add__(self, other: FVMTerm) -> FVMTerm:
        return Expr(self, other, "+")

    def __sub__(self, other: FVMTerm) -> FVMTerm:
        return Expr(self, other, "-")

    def matvec(
        self,
        octree_level: PyOctreeLevel,
        x: torch.Tensor,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        return torch.zeros_like(x)

    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        return torch.zeros(octree_level.n_cells)

    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        # NOTE: Implicit formulation across
        # cubes and levels is not yet implemented,
        # so this is currently an explicit formulation.
        dS = torch.tensor([dx[1] * dx[2], dx[0] * dx[2], dx[0] * dx[1]])
        dV = dx[0] * dx[1] * dx[2]
        n_cells_per_cube = octree_level.n_cells_per_node
        b = torch.zeros(octree_level.n_cells)
        for i, cube in enumerate(octree_level.nodes.values()):
            if cube.node_type != NodeType.LEAF:
                continue
            _slice = slice(i * n_cells_per_cube, (i + 1) * n_cells_per_cube)
            field_i = cube.old.cells[self.fieldname]
            field_f = field_i.face_average()  # TODO: slope
            U_f = cube.old.faces[self.U_f_name]
            flow_rate_f = self._flow_rate(U_f, dS)
            q_f = field_f * flow_rate_f
            b[_slice] = q_f.integrate_cell() / dV
        return b

    def _flow_rate(
        self, U_f: FaceTensor, dS: Float[torch.Tensor, " 3"]
    ) -> FaceTensor:
        return FaceTensor(
            U_f.w_interior, U_f.x * dS[0], U_f.y * dS[1], U_f.z * dS[2]
        )


class Expr(FVMTerm):
    def __init__(self, left: FVMTerm, right: FVMTerm, op: str):
        self.left = left
        self.right = right
        self.op = op

    def matvec(
        self,
        octree_level: PyOctreeLevel,
        x: torch.Tensor,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        match self.op:
            case "+":
                return self.left.matvec(
                    octree_level, x, dt, dx
                ) + self.right.matvec(octree_level, x, dt, dx)
            case "-":
                return self.left.matvec(
                    octree_level, x, dt, dx
                ) - self.right.matvec(octree_level, x, dt, dx)
            case _:
                raise ValueError(f"Unknown operator: {self.op}")

    def diag(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        return self.left.diag(octree_level, dt, dx) + self.right.diag(
            octree_level, dt, dx
        )


    def rhs(
        self,
        octree_level: PyOctreeLevel,
        dt: float,
        dx: Float[torch.Tensor, " 3"],
    ) -> torch.Tensor:
        match self.op:
            case "+":
                return self.left.rhs(octree_level, dt, dx) + self.right.rhs(
                    octree_level, dt, dx
                )
            case "-":
                return self.left.rhs(octree_level, dt, dx) - self.right.rhs(
                    octree_level, dt, dx
                )
            case _:
                raise ValueError(f"Unknown operator: {self.op}")
