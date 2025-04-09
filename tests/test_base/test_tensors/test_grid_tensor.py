from collections.abc import Callable

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite

from gridfoam._base.tensors import GridTensor, grid_tensor
from gridfoam.utils.enums import CoordinateType


@composite
def tensor_inputs(
    draw: Callable,
) -> bool | int | float | list | np.ndarray | torch.Tensor:
    """strategy for tensor inputs"""
    input_type = draw(
        st.sampled_from(
            ["bool", "int", "float", "list", "ndarray", "tensor", "grid_tensor"]
        )
    )
    shape = draw(st.tuples(st.integers(1, 5), st.integers(1, 5)))

    if input_type == "bool":
        return draw(st.booleans())
    if input_type == "int":
        return draw(st.integers(min_value=-1e3, max_value=1e3))
    if input_type == "float":
        return draw(st.floats(width=64, allow_nan=True, allow_infinity=True))

    if input_type == "list":
        elements = st.floats(width=64, allow_nan=True, allow_infinity=True)
        return draw(st.lists(elements, min_size=1, max_size=5))

    if input_type == "ndarray":
        return draw(
            arrays(
                dtype=np.float32,
                shape=shape,
                elements=st.floats(allow_nan=True, allow_infinity=True),
            )
        )

    if input_type == "tensor":
        array = draw(
            arrays(
                dtype=np.float32,
                shape=shape,
                elements=st.floats(allow_nan=True, allow_infinity=True),
            )
        )
        return torch.from_numpy(array)

    if input_type == "grid_tensor":
        array = draw(
            arrays(
                dtype=np.float32,
                shape=shape,
                elements=st.floats(allow_nan=True, allow_infinity=True),
            )
        )
        return GridTensor.from_numpy(array)


@composite
def coord_type_strategy(draw: Callable) -> CoordinateType | None:
    """strategy for CoordinateType"""
    return draw(
        st.sampled_from(
            [
                None,
                CoordinateType.GLOBAL,
                CoordinateType.LOCAL,
                CoordinateType.CELL_CENTER,
                CoordinateType.CELL_INDEX,
            ]
        )
    )


@composite
def device_strategy(draw: Callable) -> str:
    """strategy for device"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return draw(st.sampled_from(devices))


@pytest.mark.with_device
@given(
    tensor_input=tensor_inputs(),
    coord_type=coord_type_strategy(),
    device=device_strategy(),
)
@settings(deadline=None)
def test_grid_tensor_creation(
    tensor_input: bool
    | int
    | float
    | list
    | np.ndarray
    | torch.Tensor
    | GridTensor,
    coord_type: CoordinateType | None,
    device: str,
):
    """test grid_tensor function with various input patterns"""
    result = grid_tensor(tensor_input, coord_type=coord_type, device=device)

    # basic verification
    assert isinstance(result, GridTensor)
    assert result.coord_type == coord_type

    # device verification
    if hasattr(tensor_input, "device"):
        assert result.device.type == device


# test edge cases
@given(
    tensor_input=st.one_of(
        st.just([]),
        st.just(np.array([])),
    )
)
@settings(deadline=None)
def test_grid_tensor_edge_cases(tensor_input: list | np.ndarray):
    """test grid_tensor function with edge cases"""
    result = grid_tensor(tensor_input)
    assert isinstance(result, GridTensor)

    # if input is empty list or numpy array, check if the shape is appropriate
    if isinstance(tensor_input, list | np.ndarray) and len(tensor_input) == 0:
        assert result.numel() == 0


def test_invalid_input():
    """Test that invalid inputs raise appropriate exceptions."""
    with pytest.raises(TypeError):
        GridTensor([1, 2, 3])


# test class methods
@given(size=st.tuples(st.integers(0, 5), st.integers(0, 5)))
@settings(deadline=None)
def test_creation_methods(size: tuple[int, ...]):
    """test various creation methods of GridTensor."""
    # zeros
    zeros = GridTensor.zeros(size, coord_type=CoordinateType.LOCAL)
    assert zeros.shape == size
    assert torch.all(zeros.tensor() == 0)

    # ones
    ones = GridTensor.ones(size, coord_type=CoordinateType.LOCAL)
    assert ones.shape == size
    assert torch.all(ones.tensor() == 1)
    # full
    value = 42
    full = GridTensor.full(size, value, coord_type=CoordinateType.LOCAL)
    assert full.shape == size
    assert torch.all(full.tensor() == value)


@given(size=st.tuples(st.integers(0, 5), st.integers(0, 5)))
@settings(deadline=None)
def test_from_numpy(size: tuple[int, ...]):
    """test from_numpy method"""
    array = np.random.randn(*size)
    grid = GridTensor.from_numpy(array)
    assert grid.shape == size
    assert grid.ndim == len(size)
    assert np.allclose(grid.numpy(), array)


@given(size=st.tuples(st.integers(0, 5), st.integers(0, 5)))
@settings(deadline=None)
def test_from_tensor(size: tuple[int, ...]):
    """test from_tensor method"""
    tensor = torch.randn(*size)
    grid = GridTensor.from_tensor(tensor)
    assert grid.shape == size
    assert grid.ndim == len(size)
    assert torch.allclose(grid.tensor(), tensor)


def test_abs():
    """test abs method"""
    tensor = grid_tensor([1, -2, 3], coord_type=CoordinateType.GLOBAL)
    ret = abs(tensor)
    assert isinstance(ret, GridTensor)
    assert torch.equal(ret.tensor(), torch.tensor([1, 2, 3]))


def test_neg():
    """test neg method"""
    tensor = grid_tensor([1, -2, 3], coord_type=CoordinateType.GLOBAL)
    ret = -tensor
    assert isinstance(ret, GridTensor)
    assert torch.equal(ret.tensor(), torch.tensor([-1, 2, -3]))


@composite
def arrays_for_arithmetic_test(
    draw: Callable,
) -> tuple[np.ndarray, np.ndarray | float | int]:
    """Generate two arrays with the same shape"""
    shape = draw(st.tuples(st.integers(1, 5)))
    x = draw(
        arrays(
            dtype=np.float32,
            shape=shape,
            elements=st.floats(
                min_value=-10,
                max_value=10,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    y = draw(
        st.one_of(
            arrays(
                dtype=np.float32,
                shape=shape,
                elements=st.floats(
                    min_value=-10,
                    max_value=10,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            ),
            st.floats(
                min_value=-10,
                max_value=10,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.integers(min_value=-10, max_value=10),
        )
    )
    return x, y


@pytest.mark.with_device
@given(
    arrays_data=arrays_for_arithmetic_test(),
    device=device_strategy(),
)
@settings(deadline=None)
def test_add(
    arrays_data: tuple[np.ndarray, np.ndarray | float | int],
    device: str,
):
    """Test add method with hypothesis"""
    x, y = arrays_data
    x_tensor = grid_tensor(x, coord_type=CoordinateType.GLOBAL, device=device)

    if isinstance(y, float | int):
        y_tensor = y
        desired = np.array(x) + y
    else:
        y_tensor = grid_tensor(
            y, coord_type=CoordinateType.GLOBAL, device=device
        )
        desired = np.array(x) + np.array(y)

    result1 = x_tensor + y_tensor
    result2 = y_tensor + x_tensor
    assert isinstance(result1, GridTensor)
    assert isinstance(result2, GridTensor)
    np.testing.assert_array_equal(result1.numpy(), desired)
    np.testing.assert_array_equal(result2.numpy(), desired)


@pytest.mark.with_device
@given(
    arrays_data=arrays_for_arithmetic_test(),
    device=device_strategy(),
)
@settings(deadline=None)
def test_sub(
    arrays_data: tuple[np.ndarray, np.ndarray | float | int],
    device: str,
):
    """Test sub method with hypothesis"""
    x, y = arrays_data
    x_tensor = grid_tensor(x, coord_type=CoordinateType.GLOBAL, device=device)

    if isinstance(y, float | int):
        y_tensor = y
        desired = np.array(x) - y
    else:
        y_tensor = grid_tensor(
            y, coord_type=CoordinateType.GLOBAL, device=device
        )
        desired = np.array(x) - np.array(y)

    result1 = x_tensor - y_tensor
    result2 = y_tensor - x_tensor
    assert isinstance(result1, GridTensor)
    assert isinstance(result2, GridTensor)
    np.testing.assert_array_equal(result1.numpy(), desired)
    np.testing.assert_array_equal(result2.numpy(), -desired)


@pytest.mark.with_device
@given(
    arrays_data=arrays_for_arithmetic_test(),
    device=device_strategy(),
)
@settings(deadline=None)
def test_mul(
    arrays_data: tuple[np.ndarray, np.ndarray | float | int], device: str
):
    """Test mul method with hypothesis"""
    x, y = arrays_data
    x_tensor = grid_tensor(x, coord_type=CoordinateType.GLOBAL, device=device)

    if isinstance(y, float | int):
        y_tensor = y
        desired = np.array(x) * y
    else:
        y_tensor = grid_tensor(
            y, coord_type=CoordinateType.GLOBAL, device=device
        )
        desired = np.array(x) * np.array(y)

    result1 = x_tensor * y_tensor
    result2 = y_tensor * x_tensor
    assert isinstance(result1, GridTensor)
    assert isinstance(result2, GridTensor)
    np.testing.assert_array_equal(result1.numpy(), desired)
    np.testing.assert_array_equal(result2.numpy(), desired)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "x, y, desired",
    [
        (
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0],
            [4.0, 2.0, 2.0],
        ),
        (
            [4.0, 5.0, 6.0],
            2.0,
            [2.0, 2.0, 3.0],
        ),
        (
            2.0,
            [4.0, 5.0, 10.0],
            [0.0, 0.0, 0.0],
        ),
    ],
)
@given(
    device=device_strategy(),
)
@settings(deadline=None)
def test_floordiv(
    x: float | list[float],
    y: float | list[float],
    device: str,
    desired: list[float],
):
    tensor_x = grid_tensor(x, coord_type=CoordinateType.GLOBAL, device=device)
    tensor_y = grid_tensor(y, coord_type=CoordinateType.GLOBAL, device=device)
    result = tensor_x // tensor_y
    assert isinstance(result, GridTensor)
    assert torch.equal(result.tensor(), torch.tensor(desired, device=device))


@pytest.mark.with_device
@pytest.mark.parametrize(
    "x, y, desired",
    [
        (
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0],
            [4.0, 2.5, 2.0],
        ),
        (
            [4.0, 5.0, 6.0],
            2.0,
            [2.0, 2.5, 3.0],
        ),
        (
            2.0,
            [4.0, 5.0, 10.0],
            [0.5, 0.4, 0.2],
        ),
    ],
)
@given(
    device=device_strategy(),
)
@settings(deadline=None)
def test_truediv(
    x: float | list[float],
    y: float | list[float],
    desired: list[float],
    device: str,
):
    tensor_x = grid_tensor(x, coord_type=CoordinateType.GLOBAL, device=device)
    tensor_y = grid_tensor(y, coord_type=CoordinateType.GLOBAL, device=device)
    result = tensor_x / tensor_y
    assert isinstance(result, GridTensor)
    assert torch.equal(result.tensor(), torch.tensor(desired, device=device))


def test_bool_conversion():
    """test bool conversion"""
    tensor = grid_tensor(0.0, coord_type=CoordinateType.GLOBAL)
    assert not bool(tensor)
    tensor = grid_tensor(1.0, coord_type=CoordinateType.GLOBAL)
    assert bool(tensor)


def test_len_method():
    """test len method"""
    tensor = grid_tensor(
        [[1, 2, 3], [4, 5, 6]], coord_type=CoordinateType.GLOBAL
    )
    assert len(tensor) == 2


def test_size_method():
    """test size method"""
    tensor = grid_tensor(
        [[1, 2, 3], [4, 5, 6]], coord_type=CoordinateType.GLOBAL
    )
    assert tensor.size() == torch.Size([2, 3])


def test_numel_method():
    """test numel method"""
    tensor = grid_tensor(
        [[1, 2, 3], [4, 5, 6]], coord_type=CoordinateType.GLOBAL
    )
    assert tensor.numel() == 6


def test_item_method():
    """test item method"""
    tensor = grid_tensor(1.0, coord_type=CoordinateType.GLOBAL)
    assert tensor.item() == 1.0


def test_reshape_method():
    """test reshape method"""
    tensor = grid_tensor(
        [[1, 2, 3], [4, 5, 6]], coord_type=CoordinateType.GLOBAL
    )
    reshaped = tensor.reshape((3, 2))
    desired_tensor = grid_tensor(
        [[1, 2], [3, 4], [5, 6]], coord_type=CoordinateType.GLOBAL
    )
    assert torch.equal(reshaped.tensor(), desired_tensor.tensor())


def test_coordinate_type_compatibility():
    """Test operations between different coordinate types."""
    tensor1 = GridTensor.ones((2, 2), coord_type=CoordinateType.GLOBAL)
    tensor2 = GridTensor.ones((2, 2), coord_type=CoordinateType.LOCAL)

    with pytest.raises(ValueError):
        _ = tensor1 + tensor2

    with pytest.raises(ValueError):
        _ = tensor1 - tensor2

    with pytest.raises(ValueError):
        _ = tensor1 * tensor2

    with pytest.raises(ValueError):
        _ = tensor1 // tensor2

    with pytest.raises(ValueError):
        _ = tensor1 / tensor2

    with pytest.raises(ValueError):
        _ = tensor1 == tensor2

    with pytest.raises(ValueError):
        _ = tensor1 < tensor2

    with pytest.raises(ValueError):
        _ = tensor1 <= tensor2

    with pytest.raises(ValueError):
        _ = tensor1 > tensor2

    with pytest.raises(ValueError):
        _ = tensor1 >= tensor2


@given(
    arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(0, 5), st.integers(0, 5)),
    )
)
def test_numpy_interoperability(arr: np.ndarray):
    """Test numpy array conversion and interoperability."""
    grid = GridTensor.from_numpy(arr)
    np.testing.assert_array_equal(grid.numpy(), arr)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_conversion():
    """Test device conversion if CUDA is available."""
    tensor = GridTensor.ones((2, 2))

    gpu_tensor = tensor.to("cuda")
    assert gpu_tensor.device.type == "cuda"

    cpu_tensor = gpu_tensor.to("cpu")
    assert cpu_tensor.device.type == "cpu"


def test_indexing():
    """Test indexing operations."""
    tensor = GridTensor.ones((4, 4), coord_type=CoordinateType.GLOBAL)

    # basic indexing
    assert tensor[0, 0].tensor().item() == 1.0

    # slicing
    sliced = tensor[1:3, 1:3]
    assert sliced.shape == (2, 2)
    assert sliced.coord_type == CoordinateType.GLOBAL
    assert torch.equal(sliced.tensor(), tensor.tensor()[1:3, 1:3])

    # indexing with GridTensor
    index = grid_tensor([0, 1], coord_type=CoordinateType.GLOBAL)
    assert torch.equal(tensor[index].tensor(), tensor[[0, 1]].tensor())

    # setting with indexing
    tensor[0, 0] = 2.0
    assert tensor[0, 0].tensor().item() == 2.0


def test_reshape():
    """Test reshape operations."""
    tensor = GridTensor.ones((2, 4), coord_type=CoordinateType.GLOBAL)
    reshaped = tensor.reshape((4, 2))

    assert reshaped.shape == (4, 2)
    assert reshaped.coord_type == tensor.coord_type


def test_basic_gradient():
    """Basic gradient test"""
    tensor = grid_tensor(
        2.0, coord_type=CoordinateType.GLOBAL, requires_grad=True
    )

    y = tensor * 2
    y.backward()

    assert tensor.tensor().grad is not None
    assert tensor.tensor().grad.item() == 2.0


def test_detach_operation():
    """detach operation test"""
    tensor = grid_tensor(
        2.0, coord_type=CoordinateType.GLOBAL, requires_grad=True
    )

    detached = tensor.detach()

    assert isinstance(detached, GridTensor)
    assert detached.coord_type == tensor.coord_type
    assert not detached.tensor().requires_grad
    assert torch.equal(detached.tensor(), tensor.tensor())


@given(
    x=st.floats(min_value=-10.0, max_value=10.0),
    y=st.floats(min_value=-10.0, max_value=10.0),
)
def test_complex_gradient(x: float, y: float):
    """complex gradient test"""
    grid_x = grid_tensor(
        x, coord_type=CoordinateType.GLOBAL, requires_grad=True
    )
    grid_y = grid_tensor(
        y, coord_type=CoordinateType.GLOBAL, requires_grad=True
    )

    # complex operation: f(x,y) = x^2 + 2xy + y^2
    result: GridTensor = grid_x * grid_x + 2 * grid_x * grid_y + grid_y * grid_y
    result.backward()

    # compare with analytical gradient
    # \partial f / \partial x = 2x + 2y
    # \partial f / \partial y = 2x + 2y
    expected_grad_x = 2 * x + 2 * y
    expected_grad_y = 2 * x + 2 * y

    assert torch.allclose(
        grid_x.tensor().grad, torch.tensor([expected_grad_x]), rtol=1e-5
    )
    assert torch.allclose(
        grid_y.tensor().grad, torch.tensor([expected_grad_y]), rtol=1e-5
    )


def test_detach():
    """detach gradient propagation stop test"""
    tensor = grid_tensor(
        2.0, coord_type=CoordinateType.GLOBAL, requires_grad=True
    )
    intermediate: GridTensor = tensor * 2
    assert intermediate.tensor().grad_fn is not None
    detached = intermediate.detach()
    assert detached.tensor().grad_fn is None


def test_clone():
    """test clone operation"""
    tensor = grid_tensor(
        2.0, coord_type=CoordinateType.GLOBAL, requires_grad=True
    )

    cloned = tensor.clone()
    assert isinstance(cloned, GridTensor)
    assert cloned.coord_type == tensor.coord_type
    assert torch.equal(cloned.tensor(), tensor.tensor())
    assert cloned.tensor().requires_grad
    assert id(cloned) != id(tensor)


@given(
    arrays_data=arrays_for_arithmetic_test(),
    device=device_strategy(),
)
@settings(deadline=None)
def test_comparison_operations(
    arrays_data: tuple[np.ndarray, np.ndarray | float | int],
    device: str,
):
    """Test comparison operations."""
    x, y = arrays_data
    x_tensor = grid_tensor(x, coord_type=CoordinateType.GLOBAL, device=device)

    if isinstance(y, float | int):
        y_tensor = y
    else:
        y_tensor = grid_tensor(
            y, coord_type=CoordinateType.GLOBAL, device=device
        )

    np.testing.assert_array_equal((x_tensor == y_tensor).numpy(), (x == y))
    np.testing.assert_array_equal((x_tensor < y_tensor).numpy(), (x < y))
    np.testing.assert_array_equal((x_tensor <= y_tensor).numpy(), (x <= y))
    np.testing.assert_array_equal((x_tensor > y_tensor).numpy(), (x > y))
    np.testing.assert_array_equal((x_tensor >= y_tensor).numpy(), (x >= y))


def test_torch_function():
    """test torch function"""
    tensor = grid_tensor(
        [[1, 2, 3], [4, 5, 6]], coord_type=CoordinateType.GLOBAL
    )
    median = torch.median(tensor, dim=1).values
    assert torch.equal(median, torch.tensor([2, 5]))
