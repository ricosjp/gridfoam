from typing import Any

import numpy as np
import pytest
import torch

from gridfoam import GridTensor


@pytest.mark.with_device
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize(
    "data, coord_type",
    [
        (5.0, "global"),
        ([1.0, 2.0, 3.0], "local"),
        (np.array([4.0, 5.0, 6.0]), "cell"),
        (torch.tensor([7.0, 8.0, 9.0]), "global"),
    ],
)
def test_creation(data: Any, coord_type: str, requires_grad: bool, device: str):
    tensor = GridTensor(
        data, coord_type=coord_type, requires_grad=requires_grad, device=device
    )
    assert isinstance(tensor, GridTensor)
    assert tensor.coord_type == coord_type
    assert tensor.requires_grad == requires_grad
    assert str(tensor.device) == device


@pytest.mark.parametrize("coord_type", ["global", "local", "cell"])
@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
    ],
)
def test_repr(data: list[float], coord_type: str):
    tensor = GridTensor(data, coord_type=coord_type)
    assert repr(tensor) == f"GridTensor({data}) (coord_type: {coord_type})"


@pytest.mark.with_device
@pytest.mark.parametrize("coord_type", ["global", "local", "cell"])
@pytest.mark.parametrize(
    "input, slice, desired",
    [
        ([1, 2, 3, 4, 5], slice(None, 2), [1, 2]),
        ([1, 2, 3, 4, 5], slice(None, -2), [1, 2, 3]),
        ([1, 2, 3, 4, 5], slice(2, None), [3, 4, 5]),
        ([1, 2, 3, 4, 5], slice(-2, None), [4, 5]),
        ([1, 2, 3, 4, 5], slice(1, 4), [2, 3, 4]),
        ([1, 2, 3, 4, 5], slice(2, -2), [3]),
        ([1, 2, 3, 4, 5], slice(None, None, 2), [1, 3, 5]),
    ],
)
def test_getitem(
    input: Any,
    slice: Any,
    coord_type: str,
    device: str,
    desired: Any,
):
    tensor = GridTensor(input, coord_type=coord_type, device=device)
    result = tensor[slice]
    assert isinstance(result, GridTensor)
    assert result.coord_type == coord_type
    assert str(result.device) == device
    assert torch.equal(result, torch.tensor(desired, device=device))


@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3],
        GridTensor([1, 2, 3], coord_type="local"),
    ],
)
def test_unavailable_operation(other: Any):
    tensor = GridTensor([1, 2, 3], coord_type="global")

    with pytest.raises(TypeError):
        _ = tensor + other

    with pytest.raises(TypeError):
        _ = tensor - other

    with pytest.raises(TypeError):
        _ = tensor == other

    with pytest.raises(TypeError):
        _ = tensor < other

    with pytest.raises(TypeError):
        _ = tensor <= other

    with pytest.raises(TypeError):
        _ = tensor > other

    with pytest.raises(TypeError):
        _ = tensor >= other


@pytest.mark.with_device
@pytest.mark.parametrize("coord_type", ["global", "local", "cell"])
@pytest.mark.parametrize(
    "data1, data2, desired",
    [
        (5, 3, 8),
        ([1, 2, 3], [4, 5, 6], [5, 7, 9]),
        (np.array([4, 5, 6]), np.array([1, 1, 1]), [5, 6, 7]),
    ],
)
def test_add(
    data1: Any,
    data2: Any,
    coord_type: str,
    device: str,
    desired: Any,
):
    input1 = GridTensor(data1, coord_type=coord_type, device=device)
    input2 = GridTensor(data2, coord_type=coord_type, device=device)
    result = input1 + input2
    assert isinstance(result, GridTensor)
    assert result.coord_type == coord_type
    assert str(result.device) == device
    assert torch.equal(result, torch.tensor(desired, device=device))


@pytest.mark.with_device
@pytest.mark.parametrize("coord_type", ["global", "local", "cell"])
@pytest.mark.parametrize(
    "data1, data2, desired",
    [
        (5, 3, 2),
        ([1, 2, 3], [4, 5, 6], [-3, -3, -3]),
        (np.array([4, 5, 6]), np.array([1, 1, 1]), [3, 4, 5]),
    ],
)
def test_sub(
    data1: Any,
    data2: Any,
    coord_type: str,
    device: str,
    desired: Any,
):
    input1 = GridTensor(data1, coord_type=coord_type, device=device)
    input2 = GridTensor(data2, coord_type=coord_type, device=device)
    result = input1 - input2
    assert isinstance(result, GridTensor)
    assert result.coord_type == coord_type
    assert str(result.device) == device
    assert torch.equal(result, torch.tensor(desired, device=device))


@pytest.mark.with_device
@pytest.mark.parametrize("coord_type", ["global", "local", "cell"])
@pytest.mark.parametrize(
    "data, scalar, desired",
    [
        (5, 2, 10),
        ([1, 2, 3], 2, [2, 4, 6]),
        (np.array([4, 5, 6]), 3, [12, 15, 18]),
    ],
)
def test_mul(
    data: Any,
    scalar: Any,
    coord_type: str,
    device: str,
    desired: Any,
):
    input = GridTensor(data, coord_type=coord_type, device=device)
    result1 = input * scalar
    result2 = scalar * input
    assert isinstance(result1, GridTensor)
    assert isinstance(result2, GridTensor)
    assert result1.coord_type == coord_type
    assert result2.coord_type == coord_type
    assert str(result1.device) == device
    assert str(result2.device) == device
    assert torch.equal(result1, torch.tensor(desired, device=device))
    assert torch.equal(result2, torch.tensor(desired, device=device))


@pytest.mark.with_device
@pytest.mark.parametrize(
    "data1, data2, desired",
    [
        (GridTensor(5), 5, True),
        (GridTensor([1, 2, 3]), GridTensor([9, 5, 3]), [False, False, True]),
        (GridTensor([1, 2, 3]), GridTensor([-1, 5, 8]), [False, False, False]),
        (GridTensor([1, 2, 3]), GridTensor([1, 2, 0]), [True, True, False]),
        (GridTensor([1, 1, 1]), 1, [True, True, True]),
    ],
)
def test_eq(
    data1: Any,
    data2: Any,
    desired: Any,
):
    result = data1 == data2
    assert isinstance(result, GridTensor)
    assert torch.equal(result, torch.tensor(desired))


@pytest.mark.parametrize(
    "data1, data2, desired",
    [
        (GridTensor(5), 6, True),
        (GridTensor([1, 2, 3]), GridTensor([9, 5, 3]), [True, True, False]),
        (GridTensor([1, 2, 3]), GridTensor([-1, 5, 8]), [False, True, True]),
        (GridTensor([1, 2, 3]), GridTensor([1, 2, 0]), [False, False, False]),
        (GridTensor([1, 1, 1]), 1, [False, False, False]),
    ],
)
def test_lt(data1: Any, data2: Any, desired: Any):
    result = data1 < data2
    assert isinstance(result, GridTensor)
    assert torch.equal(result, torch.tensor(desired))


@pytest.mark.parametrize(
    "data1, data2, desired",
    [
        (GridTensor(5), 6, True),
        (GridTensor([1, 2, 3]), GridTensor([9, 5, 3]), [True, True, True]),
        (GridTensor([1, 2, 3]), GridTensor([-1, 5, 8]), [False, True, True]),
        (GridTensor([1, 2, 3]), GridTensor([1, 2, 0]), [True, True, False]),
        (GridTensor([1, 1, 1]), 1, [True, True, True]),
    ],
)
def test_le(data1: Any, data2: Any, desired: Any):
    result = data1 <= data2
    assert isinstance(result, GridTensor)
    assert torch.equal(result, torch.tensor(desired))


@pytest.mark.parametrize(
    "data1, data2, desired",
    [
        (GridTensor(5), 6, False),
        (GridTensor([1, 2, 3]), GridTensor([9, 5, 3]), [False, False, False]),
        (GridTensor([1, 2, 3]), GridTensor([-1, 5, 8]), [True, False, False]),
        (GridTensor([1, 2, 3]), GridTensor([1, 2, 0]), [False, False, True]),
        (GridTensor([1, 1, 1]), 1, [False, False, False]),
    ],
)
def test_gt(data1: Any, data2: Any, desired: Any):
    result = data1 > data2
    assert isinstance(result, GridTensor)
    assert torch.equal(result, torch.tensor(desired))


@pytest.mark.parametrize(
    "data1, data2, desired",
    [
        (GridTensor(5), 6, False),
        (GridTensor([1, 2, 3]), GridTensor([9, 5, 3]), [False, False, True]),
        (GridTensor([1, 2, 3]), GridTensor([-1, 5, 8]), [True, False, False]),
        (GridTensor([1, 2, 3]), GridTensor([1, 2, 0]), [True, True, True]),
        (GridTensor([1, 1, 1]), 1, [True, True, True]),
    ],
)
def test_ge(data1: Any, data2: Any, desired: Any):
    result = data1 >= data2
    assert isinstance(result, GridTensor)
    assert torch.equal(result, torch.tensor(desired))


@pytest.mark.with_device
@pytest.mark.parametrize("coord_type", ["global", "local", "cell"])
@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize(
    "data, desired",
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ],
)
def test_clone(
    data: Any,
    coord_type: str,
    requires_grad: bool,
    device: str,
    desired: Any,
):
    input = GridTensor(
        data, coord_type=coord_type, device=device, requires_grad=requires_grad
    )
    result = input.clone()
    assert isinstance(result, GridTensor)
    assert result.coord_type == coord_type
    assert str(result.device) == device
    assert torch.equal(result, torch.tensor(desired, device=device))
    assert result.requires_grad == requires_grad
    assert id(result) != id(input)


@pytest.mark.parametrize("requires_grad", [True, False])
@pytest.mark.parametrize(
    "data, src_dtype, dst_dtype",
    [
        ([1.0, 2.0, 3.0], torch.float32, torch.float64),
        ([1.0, 2.0, 3.0], torch.float64, torch.float32),
    ],
)
def test_to(
    data: Any,
    src_dtype: torch.dtype,
    dst_dtype: torch.dtype,
    requires_grad: bool,
):
    input = GridTensor(
        data,
        dtype=src_dtype,
        requires_grad=requires_grad,
    )
    result = input.to(dtype=dst_dtype)
    assert isinstance(result, GridTensor)
    assert result.dtype == dst_dtype
    assert torch.equal(result, torch.tensor(data, dtype=dst_dtype))
    assert result.requires_grad == requires_grad
