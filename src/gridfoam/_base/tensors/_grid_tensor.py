from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Self, TypeAlias

import numpy as np
import torch

from gridfoam._base.tensors._interface import IGridTensor
from gridfoam.utils.enums import CoordinateType

Number: TypeAlias = int | float | bool
ArrayLikeObject: TypeAlias = list[Number] | np.ndarray
DeviceLikeType: TypeAlias = str | torch.device | int
_HANDLED_FUNCTIONS: dict[str, Callable] = {}


def grid_tensor(
    tensor: Number | ArrayLikeObject | torch.Tensor | GridTensor,
    coord_type: CoordinateType | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> GridTensor:
    if isinstance(tensor, GridTensor):
        if (
            coord_type != tensor.coord_type
            or dtype != tensor.dtype
            or device != tensor.device
        ):
            return GridTensor(
                tensor.tensor(),
                coord_type=coord_type,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )
        return tensor
    if isinstance(tensor, int | float | bool | list | np.ndarray):
        tensor = torch.tensor(
            tensor, dtype=dtype, device=device, requires_grad=requires_grad
        )
    return GridTensor(
        tensor,
        coord_type=coord_type,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


class GridTensor(IGridTensor):
    """
    GridTensor is a wrapper of torch.Tensor with a coordinate type.

    Parameters
    ----------
    tensor (torch.Tensor):
        Tensor to be wrapped.
    coord_type (CoordinateType, optional):
        Coordinate type of the tensor.

    Examples
    --------
    >>> tensor = torch.randn(5, 100, 3)
    >>> grid_tensor = GridTensor(
    ...     tensor,
    ...     coord_type=CoordinateType.GLOBAL,
    ... )
    >>> print(grid_tensor)
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        coord_type: CoordinateType | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Expect torch.Tensor but {tensor.__class__} was fed"
            )
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        if requires_grad:
            tensor.requires_grad_()
        self._tensor = tensor
        self._coord_type = coord_type

    @classmethod
    def from_numpy(
        cls, array: np.ndarray, coord_type: CoordinateType | None = None
    ) -> GridTensor:
        tensor = torch.from_numpy(array)
        return cls(tensor, coord_type=coord_type)

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, coord_type: CoordinateType | None = None
    ) -> GridTensor:
        return cls(tensor, coord_type=coord_type)

    @classmethod
    def full(
        cls,
        size: Sequence[int],
        fill_value: Number,
        coord_type: CoordinateType | None = None,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
    ) -> GridTensor:
        tensor = torch.full(
            size,
            fill_value,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        return cls(tensor, coord_type=coord_type)

    @classmethod
    def zeros(
        cls,
        size: Sequence[int],
        coord_type: CoordinateType | None = None,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
    ) -> GridTensor:
        tensor = torch.zeros(
            size, device=device, dtype=dtype, requires_grad=requires_grad
        )
        return cls(tensor, coord_type=coord_type)

    @classmethod
    def ones(
        cls,
        size: Sequence[int],
        coord_type: CoordinateType | None = None,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool = False,
    ) -> GridTensor:
        tensor = torch.ones(
            size, device=device, dtype=dtype, requires_grad=requires_grad
        )
        return cls(tensor, coord_type=coord_type)

    @property
    def coord_type(self) -> CoordinateType | None:
        """Coordinate type of the tensor.

        Returns:
            CoordinateType | None: Coordinate type of the tensor.
        """
        return self._coord_type

    @property
    def shape(self) -> torch.Size:
        """Shape of the tensor.

        Returns:
            torch.Size: Shape of the tensor.
        """
        return self._tensor.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor.

        Returns:
            int: Number of dimensions of the tensor.
        """
        return self._tensor.ndim

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the tensor.

        Returns:
            torch.dtype: Data type of the tensor.
        """
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        """Device of the tensor.

        Returns:
            torch.device: Device of the tensor.
        """
        return self._tensor.device

    def __repr__(self) -> str:
        return (
            f"GridTensor({self._tensor}, Coordinate type: {self._coord_type})"
        )

    def __eq__(self, other: GridTensor):
        return torch.eq(self, other)

    def __lt__(self, other: GridTensor):
        return torch.lt(self, other)

    def __le__(self, other: GridTensor):
        return torch.le(self, other)

    def __gt__(self, other: GridTensor):
        return torch.gt(self, other)

    def __ge__(self, other: GridTensor):
        return torch.ge(self, other)

    def __abs__(self) -> GridTensor:
        return torch.abs(self)

    def __neg__(self) -> GridTensor:
        return -1 * self

    def __add__(self, other: GridTensor) -> GridTensor:
        return torch.add(self, other)

    def __radd__(self, other: GridTensor) -> GridTensor:
        return torch.add(self, other)

    def __sub__(self, other: GridTensor) -> GridTensor:
        return torch.sub(self, other)

    def __rsub__(self, other: GridTensor) -> GridTensor:
        return -torch.sub(self, other)

    def __mul__(self, other: GridTensor) -> GridTensor:
        return torch.mul(self, other)

    def __rmul__(self, other: GridTensor) -> GridTensor:
        return torch.mul(self, other)

    def __floordiv__(self, other: GridTensor) -> GridTensor:
        return torch.div(self, other, rounding_mode="floor")

    def __truediv__(self, other: GridTensor) -> GridTensor:
        return torch.div(self, other)

    def __rtruediv__(self, other: GridTensor) -> GridTensor:
        return torch.div(other, self)

    def __bool__(self) -> bool:
        return bool(self._tensor)

    def __getitem__(self, indices: Any) -> GridTensor:
        if isinstance(indices, GridTensor):
            indices = indices._tensor
        return GridTensor(self._tensor[indices], self.coord_type)

    def __setitem__(self, indices: Any, value: Any) -> Self:
        self._tensor[indices] = value
        return self

    def __len__(self) -> int:
        return len(self._tensor)

    def numpy(self) -> np.ndarray:
        """Convert to numpy.ndarray.

        Returns:
            numpy.ndarray: Numpy array.
        """
        return self._tensor.cpu().detach().numpy()

    def tensor(self) -> torch.Tensor:
        """Convert to torch.Tensor.

        Returns:
            torch.Tensor: Tensor.
        """
        return self._tensor

    def size(self) -> torch.Size:
        """Size of the tensor.

        Returns:
            torch.Size: Size of the tensor.
        """
        return self._tensor.size()

    def numel(self) -> int:
        """Number of elements in the tensor.

        Returns:
            int: Number of elements in the tensor.
        """
        return torch.numel(self._tensor)

    def item(self) -> Number:
        """Convert to a single element tensor.

        Returns:
            Number: Single element tensor.
        """
        return self._tensor.item()

    def reshape(
        self,
        shape: Sequence[int],
    ) -> GridTensor:
        """
        Reshape the tensor.

        Returns:
            PhlowerTensor: Reshaped tensor.
        """
        return GridTensor(
            torch.reshape(self.tensor(), shape),
            coord_type=self.coord_type,
        )

    def to(
        self,
        device: str | torch.device = None,
        non_blocking: bool = False,
        dtype: torch.dtype = None,
        coord_type: CoordinateType | None = None,
    ) -> GridTensor:
        """
        Convert the tensor to a different device or data type.

        Returns:
            GridTensor: Converted tensor.
        """
        new_tensor = self._tensor.to(
            device=device, dtype=dtype, non_blocking=non_blocking
        )
        if coord_type is None:
            coord_type = self.coord_type
        return GridTensor(
            new_tensor,
            coord_type=coord_type,
        )

    def detach(self) -> GridTensor:
        """
        Detach the tensor.

        Returns:
            GridTensor: Detached tensor.
        """
        return GridTensor(
            self._tensor.detach(),
            coord_type=self.coord_type,
        )

    def backward(self) -> None:
        """
        Backward the tensor.
        """
        self._tensor.backward()

    def clone(self) -> GridTensor:
        """
        Clone the tensor.

        Returns:
            GridTensor: Cloned tensor.
        """
        tensor = self._tensor.clone()
        return GridTensor(
            tensor,
            coord_type=self.coord_type,
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ) -> torch.Tensor | GridTensor:
        if kwargs is None:
            kwargs = {}

        # override functions for GridTensor
        if func in _HANDLED_FUNCTIONS:
            return _HANDLED_FUNCTIONS[func](*args, **kwargs)

        # other functions are not overridden so just calculate as tensor
        _tensors = _recursive_resolve(args, "_tensor")
        ret: torch.Tensor = func(*_tensors, **kwargs)
        return ret


def _recursive_resolve(
    args: Iterable | Any,
    attr: str,
) -> Any:
    if isinstance(args, tuple | list):
        return [_recursive_resolve(v, attr) for v in args]

    return getattr(args, attr, args)


def grid_tensor_wrap_implements(torch_function: Callable) -> Callable:
    """Register a torch function override for GridTensor"""

    def decorator(func: Callable) -> Callable:
        functools.update_wrapper(func, torch_function)
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@grid_tensor_wrap_implements(torch.eq)
def eq(
    input: GridTensor,
    other: GridTensor | Number,
) -> GridTensor:
    if isinstance(other, Number):
        tensor = input.tensor().eq(other)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Eq operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = torch.eq(input._tensor, other._tensor)
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.lt)
def lt(
    input: GridTensor,
    other: GridTensor | Number,
) -> GridTensor:
    if isinstance(other, Number):
        tensor = input.tensor().lt(other)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Lt operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = torch.lt(input._tensor, other._tensor)
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.le)
def le(
    input: GridTensor,
    other: GridTensor | Number,
) -> GridTensor:
    if isinstance(other, Number):
        tensor = input.tensor().le(other)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Le operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = torch.le(input._tensor, other._tensor)
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.gt)
def gt(
    input: GridTensor,
    other: GridTensor | Number,
) -> GridTensor:
    if isinstance(other, Number):
        tensor = input.tensor().gt(other)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Gt operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = torch.gt(input._tensor, other._tensor)
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.ge)
def ge(
    input: GridTensor,
    other: GridTensor | Number,
) -> GridTensor:
    if isinstance(other, Number):
        tensor = input.tensor().ge(other)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Ge operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = torch.ge(input._tensor, other._tensor)
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.abs)
def abs(
    input: GridTensor,
) -> GridTensor:
    tensor = torch.abs(input._tensor)
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.add)
def add(
    input: GridTensor,
    other: GridTensor | Number,
) -> GridTensor:
    if isinstance(other, Number):
        tensor = input.tensor().add(other)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Add operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = input._tensor + other._tensor
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.sub)
def sub(
    input: GridTensor,
    other: GridTensor | Number,
) -> GridTensor:
    if isinstance(other, Number):
        tensor = input.tensor().sub(other)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Sub operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = input._tensor - other._tensor
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.mul)
def mul(
    input: GridTensor,
    other: GridTensor | Number,
) -> GridTensor:
    if isinstance(other, Number):
        tensor = input.tensor().mul(other)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Mul operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = input._tensor * other._tensor
    return GridTensor(tensor, coord_type=input.coord_type)


@grid_tensor_wrap_implements(torch.div)
def div(
    input: GridTensor | Number,
    other: GridTensor | Number,
    rounding_mode: str | None = None,
) -> GridTensor:
    if isinstance(input, Number):
        tensor = input / other.tensor()
        return GridTensor(tensor, coord_type=other.coord_type)
    if isinstance(other, Number):
        tensor = input.tensor().div(other, rounding_mode=rounding_mode)
        return GridTensor(tensor, coord_type=input.coord_type)
    if input.coord_type != other.coord_type:
        raise ValueError(
            f"Div operation for different coordinate types is not allowed."
            f"Input: {input.coord_type}"
            f"Other: {other.coord_type}"
        )
    tensor = torch.div(
        input._tensor, other._tensor, rounding_mode=rounding_mode
    )
    return GridTensor(tensor, coord_type=input.coord_type)
