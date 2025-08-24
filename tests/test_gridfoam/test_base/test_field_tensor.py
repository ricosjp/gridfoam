import pytest
import torch

from gridfoam._base.field_tensor import FieldTensor


class TestFieldTensor:
    """Test FieldTensor class"""

    @pytest.fixture
    def vector_field_tensor(self) -> FieldTensor:
        """Create a simple FieldTensor for testing"""
        # Create a 3D tensor with shape (8, 8, 8, 3) for vector field
        raw_tensor = torch.arange(1536, dtype=torch.float32).reshape(8, 8, 8, 3)
        return FieldTensor(width=4, bnd=2, raw=raw_tensor)

    @pytest.fixture
    def scalar_field_tensor(self) -> FieldTensor:
        """Create a scalar FieldTensor for testing"""
        # Create a 3D tensor with shape (8, 8, 8) for scalar field
        raw_tensor = torch.arange(512, dtype=torch.float32).reshape(8, 8, 8)
        return FieldTensor(width=4, bnd=2, raw=raw_tensor)

    def test_field_tensor_initialization(self):
        """Test FieldTensor initialization"""
        raw_tensor = torch.randn(8, 8, 8, 3)
        field_tensor = FieldTensor(width=4, bnd=2, raw=raw_tensor)

        assert field_tensor.width == 4
        assert field_tensor.bnd == 2
        assert torch.equal(field_tensor.raw, raw_tensor)

    def test_interior_property_vector_field(
        self, vector_field_tensor: FieldTensor
    ):
        """Test interior property for vector field"""
        interior = vector_field_tensor.interior

        # Should extract the interior region (excluding boundary layers)
        # With bnd=2, interior should be
        # from index 2 to -2 in each spatial dimension
        expected_shape = (4, 4, 4, 3)  # (8-2-2, 8-2-2, 8-2-2, 3)
        assert interior.shape == expected_shape

        # Check that interior contains the expected values
        # Interior should start from index (2,2,2) in the original tensor
        expected_start_value = vector_field_tensor.raw[2, 2, 2, 0]
        assert interior[0, 0, 0, 0] == expected_start_value

    def test_interior_property_scalar_field(
        self, scalar_field_tensor: FieldTensor
    ):
        """Test interior property for scalar field"""
        interior = scalar_field_tensor.interior

        # Should extract the interior region (excluding boundary layers)
        expected_shape = (4, 4, 4)  # (8-2-2, 8-2-2, 8-2-2)
        assert interior.shape == expected_shape

        # Check that interior contains the expected values
        expected_start_value = scalar_field_tensor.raw[2, 2, 2]
        assert interior[0, 0, 0] == expected_start_value

    def test_xm_property(self, vector_field_tensor: FieldTensor):
        """Test xm (x-minus) boundary property"""
        xm = vector_field_tensor.xm

        # xm should be the left boundary in x-direction
        expected_shape = (2, 4, 4, 3)  # (bnd, width, width, channels)
        assert xm.shape == expected_shape

        # xm should contain values from x=0 to x=bnd-1
        assert torch.equal(xm, vector_field_tensor.raw[0:2, 2:6, 2:6, :])

    def test_xp_property(self, vector_field_tensor: FieldTensor):
        """Test xp (x-plus) boundary property"""
        xp = vector_field_tensor.xp

        # xp should be the right boundary in x-direction
        expected_shape = (2, 4, 4, 3)  # (bnd, width, width, channels)
        assert xp.shape == expected_shape

        # xp should contain values from x=-bnd to x=-1
        assert torch.equal(xp, vector_field_tensor.raw[6:8, 2:6, 2:6, :])

    def test_ym_property(self, vector_field_tensor: FieldTensor):
        """Test ym (y-minus) boundary property"""
        ym = vector_field_tensor.ym

        # ym should be the left boundary in y-direction
        expected_shape = (4, 2, 4, 3)  # (width, bnd, width, channels)
        assert ym.shape == expected_shape

        # ym should contain values from y=0 to y=bnd-1
        assert torch.equal(ym, vector_field_tensor.raw[2:6, 0:2, 2:6, :])

    def test_yp_property(self, vector_field_tensor: FieldTensor):
        """Test yp (y-plus) boundary property"""
        yp = vector_field_tensor.yp

        # yp should be the right boundary in y-direction
        expected_shape = (4, 2, 4, 3)  # (width, bnd, width, channels)
        assert yp.shape == expected_shape

        # yp should contain values from y=-bnd to y=-1
        assert torch.equal(yp, vector_field_tensor.raw[2:6, 6:8, 2:6, :])

    def test_zm_property(self, vector_field_tensor: FieldTensor):
        """Test zm (z-minus) boundary property"""
        zm = vector_field_tensor.zm

        # zm should be the left boundary in z-direction
        expected_shape = (4, 4, 2, 3)  # (width, width, bnd, channels)
        assert zm.shape == expected_shape

        # zm should contain values from z=0 to z=bnd-1
        assert torch.equal(zm, vector_field_tensor.raw[2:6, 2:6, 0:2, :])

    def test_zp_property(self, vector_field_tensor: FieldTensor):
        """Test zp (z-plus) boundary property"""
        zp = vector_field_tensor.zp

        # zp should be the right boundary in z-direction
        expected_shape = (4, 4, 2, 3)  # (width, width, bnd, channels)
        assert zp.shape == expected_shape

        # zp should contain values from z=-bnd to z=-1
        assert torch.equal(zp, vector_field_tensor.raw[2:6, 2:6, 6:8, :])

    def test_boundary_properties_scalar_field(
        self, scalar_field_tensor: FieldTensor
    ):
        """Test boundary properties for scalar field"""
        # Test all boundary properties for scalar field
        xm = scalar_field_tensor.xm
        xp = scalar_field_tensor.xp
        ym = scalar_field_tensor.ym
        yp = scalar_field_tensor.yp
        zm = scalar_field_tensor.zm
        zp = scalar_field_tensor.zp

        # Check shapes for scalar field (no channel dimension)
        assert xm.shape == (2, 4, 4)
        assert xp.shape == (2, 4, 4)
        assert ym.shape == (4, 2, 4)
        assert yp.shape == (4, 2, 4)
        assert zm.shape == (4, 4, 2)
        assert zp.shape == (4, 4, 2)

    def test_different_boundary_widths(self):
        """Test FieldTensor with different boundary widths"""
        # Test with bnd=1
        raw_tensor = torch.randn(8, 8, 8, 2)
        field_tensor_bnd1 = FieldTensor(width=6, bnd=1, raw=raw_tensor)

        assert field_tensor_bnd1.interior.shape == (6, 6, 6, 2)
        assert field_tensor_bnd1.xm.shape == (1, 6, 6, 2)
        assert field_tensor_bnd1.xp.shape == (1, 6, 6, 2)

        # Test with bnd=3
        raw_tensor = torch.randn(12, 12, 12, 1)
        field_tensor_bnd3 = FieldTensor(width=6, bnd=3, raw=raw_tensor)

        assert field_tensor_bnd3.interior.shape == (6, 6, 6, 1)
        assert field_tensor_bnd3.xm.shape == (3, 6, 6, 1)
        assert field_tensor_bnd3.xp.shape == (3, 6, 6, 1)

    def test_dataclass_behavior(self):
        """Test that FieldTensor behaves as a dataclass"""
        raw_tensor = torch.randn(10, 10, 10, 3)
        field_tensor = FieldTensor(width=6, bnd=2, raw=raw_tensor)

        # Test that attributes can be accessed
        assert field_tensor.width == 6
        assert field_tensor.bnd == 2
        assert torch.equal(field_tensor.raw, raw_tensor)

        # Test that attributes can be modified (dataclass is not frozen)
        field_tensor.width = 8
        assert field_tensor.width == 8

    def test_edge_cases(self):
        """Test edge cases for FieldTensor"""
        # Test with minimal tensor size
        raw_tensor = torch.randn(4, 4, 4, 1)
        field_tensor = FieldTensor(width=0, bnd=2, raw=raw_tensor)

        # Interior should be empty
        assert field_tensor.interior.shape == (0, 0, 0, 1)

        # Boundaries should still be extractable
        assert field_tensor.xm.shape == (2, 0, 0, 1)
        assert field_tensor.xp.shape == (2, 0, 0, 1)
        assert field_tensor.ym.shape == (0, 2, 0, 1)
        assert field_tensor.yp.shape == (0, 2, 0, 1)
        assert field_tensor.zm.shape == (0, 0, 2, 1)
        assert field_tensor.zp.shape == (0, 0, 2, 1)

    def test_interior_setter_vector_field(
        self, vector_field_tensor: FieldTensor
    ):
        """Test interior setter for vector field"""
        # Create a new tensor to set as interior
        new_interior = torch.ones(4, 4, 4, 3) * 999.0

        # Set the interior
        vector_field_tensor.interior = new_interior

        # Verify the interior was set correctly
        assert torch.equal(vector_field_tensor.interior, new_interior)

    def test_interior_setter_scalar_field(
        self, scalar_field_tensor: FieldTensor
    ):
        """Test interior setter for scalar field"""
        # Create a new tensor to set as interior
        new_interior = torch.ones(4, 4, 4) * 888.0

        # Set the interior
        scalar_field_tensor.interior = new_interior

        # Verify the interior was set correctly
        assert torch.equal(scalar_field_tensor.interior, new_interior)

    def test_xm_setter(self, vector_field_tensor: FieldTensor):
        """Test xm (x-minus) boundary setter"""
        # Create a new tensor to set as xm boundary
        new_xm = torch.ones(2, 4, 4, 3) * 777.0

        # Set the xm boundary
        vector_field_tensor.xm = new_xm

        # Verify the xm boundary was set correctly
        assert torch.equal(vector_field_tensor.xm, new_xm)

    def test_xp_setter(self, vector_field_tensor: FieldTensor):
        """Test xp (x-plus) boundary setter"""
        # Create a new tensor to set as xp boundary
        new_xp = torch.ones(2, 4, 4, 3) * 666.0

        # Set the xp boundary
        vector_field_tensor.xp = new_xp

        # Verify the xp boundary was set correctly
        assert torch.equal(vector_field_tensor.xp, new_xp)

    def test_ym_setter(self, vector_field_tensor: FieldTensor):
        """Test ym (y-minus) boundary setter"""
        # Create a new tensor to set as ym boundary
        new_ym = torch.ones(4, 2, 4, 3) * 555.0

        # Set the ym boundary
        vector_field_tensor.ym = new_ym

        # Verify the ym boundary was set correctly
        assert torch.equal(vector_field_tensor.ym, new_ym)

    def test_yp_setter(self, vector_field_tensor: FieldTensor):
        """Test yp (y-plus) boundary setter"""
        # Create a new tensor to set as yp boundary
        new_yp = torch.ones(4, 2, 4, 3) * 444.0

        # Set the yp boundary
        vector_field_tensor.yp = new_yp

        # Verify the yp boundary was set correctly
        assert torch.equal(vector_field_tensor.yp, new_yp)

    def test_zm_setter(self, vector_field_tensor: FieldTensor):
        """Test zm (z-minus) boundary setter"""
        # Create a new tensor to set as zm boundary
        new_zm = torch.ones(4, 4, 2, 3) * 333.0

        # Set the zm boundary
        vector_field_tensor.zm = new_zm

        # Verify the zm boundary was set correctly
        assert torch.equal(vector_field_tensor.zm, new_zm)

    def test_zp_setter(self, vector_field_tensor: FieldTensor):
        """Test zp (z-plus) boundary setter"""
        # Create a new tensor to set as zp boundary
        new_zp = torch.ones(4, 4, 2, 3) * 222.0

        # Set the zp boundary
        vector_field_tensor.zp = new_zp

        # Verify the zp boundary was set correctly
        assert torch.equal(vector_field_tensor.zp, new_zp)

    def test_setter_preserves_other_regions(
        self, vector_field_tensor: FieldTensor
    ):
        """Test that setters only modify the intended region and preserve others"""
        # Store original values
        original_interior = vector_field_tensor.interior.clone()
        original_xm = vector_field_tensor.xm.clone()
        original_xp = vector_field_tensor.xp.clone()

        # Set only the interior
        new_interior = torch.ones_like(original_interior) * 999.0
        vector_field_tensor.interior = new_interior

        # Verify interior was changed
        assert torch.equal(vector_field_tensor.interior, new_interior)

        # Verify other regions were not affected
        assert torch.equal(vector_field_tensor.xm, original_xm)
        assert torch.equal(vector_field_tensor.xp, original_xp)
