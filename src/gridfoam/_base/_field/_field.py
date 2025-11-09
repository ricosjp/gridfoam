
from gridfoam._base._field._descripter import FieldDescriptor
from gridfoam._base._field._tensor import CellTensor, CubeTensor, FaceTensor
from gridfoam.config import CubeConfig
from gridfoam.utils.enums import FieldLayout


class Field:
    _cube_config: CubeConfig
    """
    A field container for managing cell / face / cube tensors.

    This class provides a structured way to manage multiple
    cell-centered / face-centered / cube-centered tensors
    for finite volume computations. It acts as
    a container that can hold multiple named tensors of different types.
    """
    @classmethod
    def set_cube_config(cls, cube_config: CubeConfig) -> None:
        """
        Set the cube configuration.
        Parameters
        ----------
        cube_config : CubeConfig
            Cube configuration.
        """
        cls._cube_config = cube_config

    def __init__(self) -> None:
        """
        Initialize the field container.
        """
        self._N = self._cube_config.interior_width
        self._H = self._cube_config.halo_width
        self._cells: dict[str, CellTensor] = {}
        self._faces: dict[str, FaceTensor] = {}
        self._cubes: dict[str, CubeTensor] = {}

    @property
    def cells(self) -> dict[str, CellTensor]:
        return self._cells

    @property
    def faces(self) -> dict[str, FaceTensor]:
        return self._faces

    @property
    def cubes(self) -> dict[str, CubeTensor]:
        return self._cubes

    def _add_cell_tensor(self, fd: FieldDescriptor) -> None:
        """
        Add a cell-centered tensor to the cube.

        Parameters
        ----------
        fd : FieldDescriptor
            Field descriptor.
        """
        N = self._N
        H = self._H
        C = fd.channels
        self._cells[fd.canonical_name] = CellTensor(N, H, C, fd.dtype, fd.device)

    def _add_face_tensor(self, fd: FieldDescriptor) -> None:
        """
        Add a face-centered tensor to the cube.

        Parameters
        ----------
        fd : FieldDescriptor
            Field descriptor.
        """
        N = self._N
        C = fd.channels
        self._faces[fd.canonical_name] = FaceTensor(N, C, fd.dtype, fd.device)

    def _add_cube_tensor(self, fd: FieldDescriptor) -> None:
        """
        Add a cube-centered tensor to the cube.

        Parameters
        ----------
        fd : FieldDescriptor
            Field descriptor.
        """
        C = fd.channels
        self._cubes[fd.canonical_name] = CubeTensor(C, fd.dtype, fd.device)

    def add_tensor(self, fd: FieldDescriptor) -> None:
        """
        Add a tensor to the field.
        Parameters
        ----------
        fd : FieldDescriptor
            Field descriptor.
        """
        match fd.layout:
            case FieldLayout.CELL:
                self._add_cell_tensor(fd)
            case FieldLayout.FACE:
                self._add_face_tensor(fd)
            case FieldLayout.CUBE:
                self._add_cube_tensor(fd)
