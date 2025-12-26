import torch

from gridfoam.DNA.config import CubeConfig
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.fielddata import CellField, FaceField, FVMatrix
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta


class CubeField:
    """
    A field container for managing cell / face tensors.

    This class provides a structured way to manage multiple
    cell-centered / face-centered tensors
    for finite volume computations. It acts as
    a container that can hold multiple named tensors of different types.
    """

    def __init__(self, cube_config: CubeConfig) -> None:
        """
        Initialize the field container.
        Parameters
        ----------
        cube_config : CubeConfig
            Cube configuration.
        """
        self._N = cube_config.interior_width
        self._H = cube_config.halo_width
        self._device = cube_config.device
        self._cells: dict[str, CellField] = {}
        self._faces: dict[str, FaceField] = {}
        self._fvmatrices: dict[str, FVMatrix] = {}

    @property
    def cells(self) -> dict[str, CellField]:
        return self._cells

    @property
    def faces(self) -> dict[str, FaceField]:
        return self._faces

    @property
    def fvmatrices(self) -> dict[str, FVMatrix]:
        return self._fvmatrices

    @property
    def N(self) -> int:
        return self._N

    @property
    def H(self) -> int:
        return self._H

    @property
    def device(self) -> torch.device:
        return self._device

    def _add_cell_tensor(self, fm: FieldMeta) -> None:
        """
        Add a cell-centered tensor to the cube.

        Parameters
        ----------
        fm : FieldMeta
            Field meta.
        """
        T = fm.time_levels
        C = fm.components
        N = self._N
        H = self._H
        self._cells[fm.name] = CellField(T, C, N, H, fm.dtype, self._device)

    def _add_face_tensor(self, fm: FieldMeta) -> None:
        """
        Add a face-centered tensor to the cube.

        Parameters
        ----------
        fm : FieldMeta
            Field meta.
        """
        T = fm.time_levels
        C = fm.components
        N = self._N
        self._faces[fm.name] = FaceField(T, C, N, fm.dtype, self._device)

    def add_equation(self, em: EquationMeta) -> None:
        """
        Add an equation to the field.

        Parameters
        ----------
        em : EquationMeta
            Equation meta.
        """
        C = em.target_field.components
        N = self._N
        H = self._H
        dtype = em.target_field.dtype
        device = self._device
        self._fvmatrices[em.name] = FVMatrix(C, N, H, dtype, device)

    def add_field(self, fm: FieldMeta) -> None:
        """
        Add a field to the cube.
        Parameters
        ----------
        fm : FieldMeta
            Field meta.
        """
        match fm.layout:
            case FieldLayout.CELL:
                self._add_cell_tensor(fm)
            case FieldLayout.FACE:
                self._add_face_tensor(fm)

    def get_field(self, fm: FieldMeta) -> CellField | FaceField:
        """
        Get a field from the cube.
        """
        match fm.layout:
            case FieldLayout.CELL:
                return self._cells[fm.name]
            case FieldLayout.FACE:
                return self._faces[fm.name]
        raise ValueError(f"Invalid field layout: {fm.layout}")

    def get_fvmatrix(self, em: EquationMeta) -> FVMatrix:
        """
        Get the FV matrix for an equation.
        """
        return self._fvmatrices[em.name]
