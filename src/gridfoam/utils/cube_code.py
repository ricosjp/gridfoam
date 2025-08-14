from gridfoam.utils.annotated_type import CubeCode
from gridfoam.utils.enums import Constants
from gridfoam.utils.morton import get_child_codes, get_parent_code


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
        raise ValueError(f"Depth must be less than {Constants.MAX_OCTREE_DEPTH}")
    root_code, morton_code = parse_cube_code(cube_code)
    child_codes = get_child_codes(morton_code, depth)
    return [
        gen_cube_code(root_code, child_code)
        for child_code in child_codes
    ]


def parent_cube_code(child_code: CubeCode, depth: int) -> CubeCode:
    if depth == 0:
        raise ValueError("Depth must be greater than 0")
    root_code, morton_code = parse_cube_code(child_code)
    parent_code = get_parent_code(morton_code, depth)
    return gen_cube_code(root_code, parent_code)
