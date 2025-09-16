import pathlib

import torch

from gridfoam import TensorGrid, save_grid
from gridfoam._simulator import grad_p, init_p


def calculate_grad_p_DrivAer(config_path: pathlib.Path):
    """Test the grid generation process."""
    grid = TensorGrid.build(config_path)
    print("grid building done")

    grid.add_field("p", (1,), torch.float32)
    grid.add_field("grad_p", (3,), torch.float32)
    grid.allocate_field_tensors()
    print("field tensors allocated")

    init_p(grid)
    print("init_p done")
    grad_p(grid)
    print("grad_p done")

    save_grid(grid)
    print("save_grid done")


if __name__ == "__main__":
    config_path = pathlib.Path("tests/data/yaml/DrivAer.yaml")
    calculate_grad_p_DrivAer(config_path)
