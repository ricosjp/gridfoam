from collections.abc import ItemsView

import pytest
import torch
from src.gridfoam._base.field_data import FieldData


class TestFieldData:
    """Test cases for FieldData class."""

    def test_init(self):
        """Test FieldData initialization."""
        device = torch.device("cpu")
        field_data = FieldData(device)

        assert field_data.device == device
        assert field_data._field_data == {}


    def test_add_multiple_fields(self):
        """Test adding multiple fields."""
        device = torch.device("cpu")
        field_data = FieldData(device)

        field_data.add_field("pressure", (10,), torch.float32)
        field_data.add_field("velocity", (10, 3), torch.float32)
        field_data.add_field("M", (10, 5, 5), torch.float64)

        assert "pressure" in field_data._field_data
        assert "velocity" in field_data._field_data
        assert "M" in field_data._field_data
        assert len(field_data._field_data) == 3

        assert field_data.pressure.shape == (10,)
        assert field_data.pressure.dtype == torch.float32
        assert field_data.velocity.shape == (10, 3)
        assert field_data.velocity.dtype == torch.float32
        assert field_data.M.shape == (10, 5, 5)
        assert field_data.M.dtype == torch.float64

    def test_add_single_value_field(self):
        """Test adding a single value field."""
        device = torch.device("cpu")
        field_data = FieldData(device)

        field_data.add_field("x", (), torch.float32)

        assert "x" in field_data._field_data
        assert len(field_data._field_data) == 1
        assert field_data.x.shape == ()
        assert field_data.x.dtype == torch.float32

    def test_getitem(self):
        """Test accessing fields using __getitem__."""
        device = torch.device("cpu")
        field_data = FieldData(device)

        field_data.add_field("density", (4,), torch.float32)

        # Test __getitem__ access
        tensor = field_data["density"]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4,)
        assert tensor.dtype == torch.float32

    def test_getitem_nonexistent_field(self):
        """Test __getitem__ with non-existent field raises AttributeError."""
        device = torch.device("cpu")
        field_data = FieldData(device)

        with pytest.raises(AttributeError):
            _ = field_data["nonexistent"]

    def test_items(self):
        """Test items() method returns correct dictionary."""
        device = torch.device("cpu")
        field_data = FieldData(device)

        field_data.add_field("u", (2, 3), torch.float32)
        field_data.add_field("v", (2, 3), torch.float32)

        items = field_data.items()

        assert isinstance(items, ItemsView)
        assert len(items) == 2

        item_list = list(items)
        assert item_list[0][0] == "u"
        assert item_list[1][0] == "v"
        assert isinstance(item_list[0][1], torch.Tensor)
        assert isinstance(item_list[1][1], torch.Tensor)
        assert item_list[0][1].shape == (2, 3)
        assert item_list[1][1].shape == (2, 3)

    def test_items_empty(self):
        """Test items() method with no fields."""
        device = torch.device("cpu")
        field_data = FieldData(device)

        items = field_data.items()

        assert isinstance(items, ItemsView)
        assert len(items) == 0

    def test_cuda_device(self):
        """Test FieldData with CUDA device if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        field_data = FieldData(device)

        field_data.add_field("test_field", (3, 3), torch.float32)

        assert field_data.device == device
        assert field_data.test_field.device.type == device.type
