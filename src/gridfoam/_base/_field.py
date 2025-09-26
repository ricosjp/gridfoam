import torch

from gridfoam._base._cell_tensor import CellTensor
from gridfoam._base._face_tensor import FaceTensor


class Field:
    """
    A field container for managing cell and face tensors.

    This class provides a structured way to manage multiple cell-centered
    and face-centered tensors for finite volume computations. It acts as
    a container that can hold multiple named tensors of different types.

    Parameters
    ----------
    w_interior : int
        Width of the interior region for cell tensors.
    w_halo : int
        Width of the halo region for cell tensors.
    device : torch.device
        Device where tensors will be allocated.
    """

    def __init__(self, w_interior: int, w_halo: int, device: torch.device):
        self.w_interior = w_interior
        self.w_halo = w_halo
        self.device = device
        self.cells: dict[str, CellTensor] = {}
        self.faces: dict[str, FaceTensor] = {}

    def add_cell_tensor(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """
        Add a cell-centered tensor to the field.

        Parameters
        ----------
        name : str
            Name of the tensor field.
        shape : tuple[int, ...]
            Shape of the field data (0-2 dimensions).
        dtype : torch.dtype
            Data type of the tensor.
        """
        self.cells[name] = CellTensor.init(
            self.w_interior, self.w_halo, shape, dtype, self.device
        )

    def add_face_tensor(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """
        Add a face-centered tensor to the field.

        Parameters
        ----------
        name : str
            Name of the tensor field.
        shape : tuple[int, ...]
            Shape of the field data (0-2 dimensions).
        dtype : torch.dtype
            Data type of the tensor.
        """
        self.faces[name] = FaceTensor.init(
            self.w_interior, shape, dtype, self.device
        )
