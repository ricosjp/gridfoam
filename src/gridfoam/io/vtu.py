import numpy as np
import pyvista as pv

from gridfoam.core.grid.base import GridBase


def to_unstructured_grid(grid: GridBase) -> pv.UnstructuredGrid:
    """
    Converts a GridBase into a UnstructuredGrid of Hexahedrons.

    Parameters
    ----------
    grid : GridBase
        The GridBase to convert.

    Returns
    -------
    pv.UnstructuredGrid
        The generated UnstructuredGrid.
    """
    n_cells = grid.num_cells

    cell_sizes = grid.cell_sizes.cpu().numpy()
    cell_centers = grid.cell_centers.cpu().numpy()

    hx = cell_sizes[:, 0] / 2.0
    hy = cell_sizes[:, 1] / 2.0
    hz = cell_sizes[:, 2] / 2.0
    cx = cell_centers[:, 0]
    cy = cell_centers[:, 1]
    cz = cell_centers[:, 2]

    # VTK Hexahedron node ordering
    # 0: -x, -y, -z
    # 1: +x, -y, -z
    # 2: +x, +y, -z
    # 3: -x, +y, -z
    # 4: -x, -y, +z
    # 5: +x, -y, +z
    # 6: +x, +y, +z
    # 7: -x, +y, +z
    pts = np.empty((n_cells, 8, 3), dtype=np.float64)

    pts[:, 0, 0] = cx - hx
    pts[:, 0, 1] = cy - hy
    pts[:, 0, 2] = cz - hz
    pts[:, 1, 0] = cx + hx
    pts[:, 1, 1] = cy - hy
    pts[:, 1, 2] = cz - hz
    pts[:, 2, 0] = cx + hx
    pts[:, 2, 1] = cy + hy
    pts[:, 2, 2] = cz - hz
    pts[:, 3, 0] = cx - hx
    pts[:, 3, 1] = cy + hy
    pts[:, 3, 2] = cz - hz

    pts[:, 4, 0] = cx - hx
    pts[:, 4, 1] = cy - hy
    pts[:, 4, 2] = cz + hz
    pts[:, 5, 0] = cx + hx
    pts[:, 5, 1] = cy - hy
    pts[:, 5, 2] = cz + hz
    pts[:, 6, 0] = cx + hx
    pts[:, 6, 1] = cy + hy
    pts[:, 6, 2] = cz + hz
    pts[:, 7, 0] = cx - hx
    pts[:, 7, 1] = cy + hy
    pts[:, 7, 2] = cz + hz

    points = pts.reshape(-1, 3)

    # Connectivity array for PyVista: [n_points, p0, p1, p2, p3, p4, p5, p6, p7]
    cells = np.empty((n_cells, 9), dtype=np.int64)
    cells[:, 0] = 8
    cells[:, 1:] = np.arange(n_cells * 8).reshape(n_cells, 8)

    cell_types = np.full(n_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

    ugrid = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    return ugrid


def update_export_cell_data(
    ugrid: pv.UnstructuredGrid,
    grid: GridBase,
) -> None:
    """
    Update ``cell_data`` from exported cell fields on the grid.

    Parameters
    ----------
    ugrid : pv.UnstructuredGrid
        Target unstructured grid to update.
    grid : GridBase
        Source grid that owns registered fields.
    """
    for field_name in grid.cellfield_names():
        field = grid.get_cellfield(field_name)
        if field is None or not field.export:
            continue
        values = field.data.detach().cpu().numpy()
        if field.tensor_rank > 1:
            # VTK stores tensor components in a flat tuple per cell.
            values = values.reshape(grid.num_cells, field.num_components)
        ugrid.cell_data[field.name] = values


def save_export_fields_as_vtu(
    grid: GridBase,
    output_path: str,
    *,
    ugrid: pv.UnstructuredGrid | None = None,
) -> pv.UnstructuredGrid:
    """
    Save all exported cell fields to a VTU file.

    Parameters
    ----------
    grid : GridBase
        Source grid to export.
    output_path : str
        Output VTU file path.
    ugrid : pv.UnstructuredGrid | None, optional
        Existing unstructured grid instance. If omitted, a new grid is built.

    Returns
    -------
    pv.UnstructuredGrid
        Unstructured grid that was updated and saved.
    """
    target_grid = ugrid if ugrid is not None else to_unstructured_grid(grid)
    update_export_cell_data(target_grid, grid)
    target_grid.save(output_path)
    return target_grid
