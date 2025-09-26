import torch

from gridfoam._base._cell_tensor import CellTensor
from gridfoam._base._face_tensor import FaceTensor


class Field:
    def __init__(self, w_interior: int, w_halo: int, device: torch.device):
        self.w_interior = w_interior
        self.w_halo = w_halo
        self.device = device
        self.cells: dict[str, CellTensor] = {}
        self.faces: dict[str, FaceTensor] = {}

    def add_cell_tensor(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        self.cells[name] = CellTensor.init(
            self.w_interior, self.w_halo, shape, dtype, self.device
        )

    def add_face_tensor(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        self.faces[name] = FaceTensor.init(
            self.w_interior, shape, dtype, self.device
        )
