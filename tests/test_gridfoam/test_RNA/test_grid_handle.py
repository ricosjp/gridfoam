import pathlib

import pytest
import torch

from gridfoam.DNA.enum import FieldLayout, FieldRole
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.RNA.builtins.builtin_fields import builtin_T
from gridfoam.RNA.defaults import default_registry
from gridfoam.RNA.grid_handle import GridHandle, _generate_grid_indices
from gridfoam.RNA.registry import SimulationMetaRegistry


def test__generate_grid_indices():
    """Test _generate_grid_indices function."""
    divisions = torch.tensor([10, 10, 10])
    indices = _generate_grid_indices(divisions)
    X = indices[0]
    Y = indices[1]
    Z = indices[2]

    expected_X = torch.arange(10).repeat(100)
    torch.testing.assert_close(X, expected_X)

    expected_Y = torch.arange(10).repeat_interleave(10).repeat(10)
    torch.testing.assert_close(Y, expected_Y)

    expected_Z = torch.arange(10).repeat_interleave(100)
    torch.testing.assert_close(Z, expected_Z)


@pytest.fixture
def grid_handle() -> GridHandle:
    configpath = pathlib.Path("tests/data/yaml/bunny.yaml")
    return GridHandle(configpath)


def test_grid_handle_properties(grid_handle: GridHandle) -> None:
    """Test GridHandle properties."""
    assert grid_handle.grid is not None
    assert grid_handle.mesh is not None
    assert grid_handle.config is not None


def test_grid_handle_iter_levels(grid_handle: GridHandle) -> None:
    """Test GridHandle.iter_levels method."""
    levels = list(grid_handle.iter_levels())
    assert len(levels) == 3
    assert levels[0].depth == 0
    assert levels[1].depth == 1
    assert levels[2].depth == 2


def test_grid_handle_iter_leaf_on_level(grid_handle: GridHandle) -> None:
    """Test GridHandle.iter_leaf_on_level method."""
    level = list(grid_handle.iter_levels())[2]
    leaves = list(grid_handle.iter_leaf_on_level(level))
    assert len(leaves) == 512


def test_grid_handle_iter_gfp_on_level(grid_handle: GridHandle) -> None:
    """Test GridHandle.iter_gfp_on_level method."""
    level = list(grid_handle.iter_levels())[1]
    gfps = list(grid_handle.iter_gfp_on_level(level))
    assert len(gfps) == 0


def test_grid_handle_iter_gfc_on_level(grid_handle: GridHandle) -> None:
    """Test GridHandle.iter_gfc_on_level method."""
    level = list(grid_handle.iter_levels())[0]
    gfcs = list(grid_handle.iter_gfc_on_level(level))
    assert len(gfcs) == 0


def test_grid_handle_iter_all_leaves(grid_handle: GridHandle) -> None:
    """Test GridHandle.iter_all_leaves method."""
    leaves = list(grid_handle.iter_all_leaves())
    assert len(leaves) == 512


@pytest.fixture
def registry():
    registry = default_registry()
    registry.register_field(builtin_T())
    return registry


def test_grid_handle_allocate_by_registry(
    grid_handle: GridHandle, registry: SimulationMetaRegistry
) -> None:
    """Test GridHandle.allocate_by_registry method."""
    grid_handle.allocate_by_registry(registry)
    for level in grid_handle.iter_levels():
        for cube in level.nodes.values():
            assert cube.field.cells["T"] is not None


def test_grid_handle_allocate_field(
    grid_handle: GridHandle, registry: SimulationMetaRegistry
) -> None:
    """Test GridHandle.allocate_field method."""
    grid_handle.allocate_by_registry(registry)
    field_meta = FieldMeta(
        name="rho",
        label="Density",
        role=FieldRole.STATE,
        layout=FieldLayout.CELL,
        components=1,
    )
    grid_handle.allocate_field(field_meta)
    for level in grid_handle.iter_levels():
        for cube in level.nodes.values():
            assert cube.field.cells["rho"] is not None


def test_grid_handle_allocate_equation(
    grid_handle: GridHandle, registry: SimulationMetaRegistry
) -> None:
    """Test GridHandle.allocate_equation method."""
    grid_handle.allocate_by_registry(registry)
    T = registry.get_field("T")
    equation_meta = EquationMeta(
        name="heat_diffusion",
        target_field=T,
        boundary_conditions=[],
        ast_root=T,
    )
    grid_handle.allocate_equation(equation_meta)
    for level in grid_handle.iter_levels():
        for cube in level.nodes.values():
            assert cube.field.fvmatrices["heat_diffusion"] is not None


def test_grid_handle_update_fvmatrix(
    grid_handle: GridHandle, registry: SimulationMetaRegistry
) -> None:
    """Test GridHandle.update_fvmatrix method."""
    grid_handle.allocate_by_registry(registry)
    T = registry.get_field("T")
    equation_meta = EquationMeta(
        name="heat_diffusion",
        target_field=T,
        boundary_conditions=[],
        ast_root=T,
    )
    grid_handle.update_fvmatrix(equation_meta)

    for level in grid_handle.iter_levels():
        for cube in level.nodes.values():
            assert cube.field.fvmatrices["heat_diffusion"] is not None


def test_grid_handle_sync_halo(
    grid_handle: GridHandle, registry: SimulationMetaRegistry
) -> None:
    """Test GridHandle.sync_halo method."""
    from gridfoam.DNA.enum import Axis

    grid_handle.allocate_by_registry(registry)
    T = registry.get_field("T")

    # Set some values in interior regions to test synchronization
    for level in grid_handle.iter_levels():
        for cube in grid_handle.iter_leaf_on_level(level):
            T_field = cube.field.cells["T"]
            # Set interior to a known value
            T_field.interior[0, 0, :, :, :] = 1.0

    # Synchronize halo
    grid_handle.sync_halo([T])

    # Check that halo regions have been synchronized from neighbors
    # For leaf cubes with neighbors,
    # halo should contain neighbor's interior values
    for level in grid_handle.iter_levels():
        for cube in grid_handle.iter_leaf_on_level(level):
            T_field = cube.field.cells["T"]
            # Halo should be finite (synchronized from neighbors or initialized)
            halo_x_forward = T_field.get_halo_along(Axis.X, forward=True)
            halo_x_backward = T_field.get_halo_along(Axis.X, forward=False)
            assert torch.all(torch.isfinite(halo_x_forward))
            assert torch.all(torch.isfinite(halo_x_backward))


def test_grid_handle_sync_gfp(
    grid_handle: GridHandle, registry: SimulationMetaRegistry
) -> None:
    """Test GridHandle.sync_gfp method."""
    grid_handle.allocate_by_registry(registry)
    T = registry.get_field("T")

    # Set values in parent cubes to test interpolation
    for level in grid_handle.iter_levels():
        if level.depth == 0:
            # Set parent (depth 0) interior values
            for cube in level.nodes.values():
                T_field = cube.field.cells["T"]
                T_field.interior[0, 0, :, :, :] = 2.0

    # Synchronize ghost from parent (should interpolate parent data to children)
    grid_handle.sync_gfp([T])

    # Check that ghost cubes (children of parents) have interpolated data
    for level in grid_handle.iter_levels():
        if level.depth > 0:
            for cube in grid_handle.iter_gfp_on_level(level):
                T_field = cube.field.cells["T"]
                # Interior should be finite (interpolated from parent)
                assert torch.all(torch.isfinite(T_field.interior[0]))
                # Should have some non-zero values (interpolated from parent)
                assert not torch.allclose(
                    T_field.interior[0, 0],
                    torch.zeros_like(T_field.interior[0, 0]),
                    atol=1e-6,
                )


def test_grid_handle_sync_gfc(
    grid_handle: GridHandle, registry: SimulationMetaRegistry
) -> None:
    """Test GridHandle.sync_gfc method."""
    grid_handle.allocate_by_registry(registry)
    T = registry.get_field("T")

    # Set values in child cubes to test coarsening
    for level in grid_handle.iter_levels():
        if level.depth == grid_handle.grid.max_depth - 1:
            # Set child (max depth) interior values
            for cube in grid_handle.iter_leaf_on_level(level):
                T_field = cube.field.cells["T"]
                T_field.interior[0, 0, :, :, :] = 3.0

    # Synchronize ghost from child (should coarsen child data to parents)
    grid_handle.sync_gfc([T])

    # Check that ghost cubes (parents of children) have coarsened data
    for level in grid_handle.iter_levels():
        if level.depth < grid_handle.grid.max_depth - 1:
            for cube in grid_handle.iter_gfc_on_level(level):
                T_field = cube.field.cells["T"]
                # Interior should be finite (coarsened from children)
                assert torch.all(torch.isfinite(T_field.interior[0]))
                # Should have some non-zero values (coarsened from children)
                # Note: coarsened values may be different from original
                # due to averaging
                assert not torch.allclose(
                    T_field.interior[0, 0],
                    torch.zeros_like(T_field.interior[0, 0]),
                    atol=1e-6,
                )


def test_grid_handle_get_dx_at_depth(grid_handle: GridHandle) -> None:
    """Test GridHandle.get_dx_at_depth method."""
    dx = grid_handle.get_dx_at_depth(0)
    assert dx[0] == torch.tensor(2.0 / 8.0)
    assert dx[1] == torch.tensor(2.0 / 8.0)
    assert dx[2] == torch.tensor(2.0 / 8.0)
    dx = grid_handle.get_dx_at_depth(1)
    assert dx[0] == torch.tensor(2.0 / 16.0)
    assert dx[1] == torch.tensor(2.0 / 16.0)
    assert dx[2] == torch.tensor(2.0 / 16.0)
