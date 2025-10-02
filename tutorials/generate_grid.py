import pathlib

import torch

from gridfoam import TensorGrid, save_grid


def generate_grid_DrivAer(config_path: pathlib.Path):
    """Generate the grid for DrivAer."""
    grid = TensorGrid.build(config_path)
    print("grid building done")

    grid.add_cell_field("U", (3,), torch.float32)
    grid.add_cell_field("T", (1,), torch.float32)
    grid.allocate_field_tensors()
    print("allocate field tensors done")

    save_grid(grid, "DrivAer.vtkhdf")
    print("save grid done")


if __name__ == "__main__":
    config_path = pathlib.Path("tests/data/yaml/DrivAer.yaml")
    generate_grid_DrivAer(config_path)
