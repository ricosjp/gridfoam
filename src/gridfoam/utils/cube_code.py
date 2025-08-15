import torch
from jaxtyping import Int32

from gridfoam.utils.annotated_type import CubeCode
from gridfoam.utils.enums import Constants
from gridfoam.utils.index import ravel_index_3d, unravel_index_3d
from gridfoam.utils.morton import (
    get_child_codes,
    get_parent_code,
    local_index_to_morton_code,
    morton_code_to_local_index,
)


def _extract_bits(cube_code: int, start: int, length: int) -> int:
    if start < 0 or length <= 0:
        raise ValueError("start must be >= 0 and length must be > 0")

    mask = (1 << length) - 1
    return (cube_code >> start) & mask


def gen_cube_code(root_code: int, morton_code: int) -> CubeCode:
    root_code_bit = root_code << Constants.MORTON_CODE_BIT_LENGTH
    return root_code_bit | morton_code


def parse_cube_code(cube_code: CubeCode) -> tuple[int, int]:
    morton_code = _extract_bits(cube_code, 0, Constants.MORTON_CODE_BIT_LENGTH)
    root_code = _extract_bits(
        cube_code,
        Constants.MORTON_CODE_BIT_LENGTH,
        Constants.ROOT_CODE_BIT_LENGTH,
    )
    return root_code, morton_code


def child_cube_codes(cube_code: CubeCode, depth: int) -> list[CubeCode]:
    if depth == Constants.MAX_OCTREE_DEPTH:
        raise ValueError(
            f"Depth must be less than {Constants.MAX_OCTREE_DEPTH}"
        )
    root_code, morton_code = parse_cube_code(cube_code)
    child_codes = get_child_codes(morton_code, depth)
    return [gen_cube_code(root_code, child_code) for child_code in child_codes]


def parent_cube_code(child_code: CubeCode, depth: int) -> CubeCode:
    if depth == 0:
        raise ValueError("Depth must be greater than 0")
    root_code, morton_code = parse_cube_code(child_code)
    parent_code = get_parent_code(morton_code, depth)
    return gen_cube_code(root_code, parent_code)


def global_indices_to_codes(
    global_indices: Int32[torch.Tensor, "n_index 3"],
    block_divisions: Int32[torch.Tensor, " 3"],
    depth: int,
) -> list[CubeCode]:
    octree_size = 1 << depth
    root_indices = global_indices // octree_size
    local_indices = global_indices % octree_size

    root_codes = ravel_index_3d(root_indices, block_divisions)
    morton_codes = local_index_to_morton_code(local_indices, depth)
    cube_codes = []
    for root_code, morton_code in zip(root_codes, morton_codes, strict=True):
        cube_codes.append(gen_cube_code(root_code.item(), morton_code.item()))
    return cube_codes


def code_to_global_index(
    cube_code: CubeCode,
    block_divisions: Int32[torch.Tensor, " 3"],
    depth: int,
) -> Int32[torch.Tensor, " 3"]:
    octree_size = 1 << depth
    root_code, morton_code = parse_cube_code(cube_code)
    root_indices = unravel_index_3d(root_code, block_divisions)
    local_indices = morton_code_to_local_index(morton_code, depth)
    return root_indices * octree_size + local_indices
