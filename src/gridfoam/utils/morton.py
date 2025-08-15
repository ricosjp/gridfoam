import torch
from jaxtyping import Int32, Int64

from gridfoam.utils.enums import Constants


def part1by2(n: Int32[torch.Tensor, "..."]) -> Int64[torch.Tensor, "..."]:
    """
    Insert two 0 bits after each of the 21 low bits of x
    """
    # mask 20+1 bits (1 bit for safety)
    n = n.to(torch.int64)
    n = n & 0x1FFFFF
    # split into high and low to prevent overflow
    # n = (n | (n << 32)) & 0x 001f 0000 0000 ffff
    high = n >> 16
    low = n & 0xFFFF

    # n = (n | (n << 16)) & 0x 001f 0000 ff00 00ff
    low = (low | (low << 16)) & 0xFF0000FF

    # n = (n | (n <<  8)) & 0x 100f 00f0 0f00 f00f
    high = (high | (high << 8)) & 0x100F
    low = (low | (low << 8)) & 0x00F00F00F00F

    # n = (n | (n <<  4)) & 0x 10c3 0c30 c30c 30c3
    high = (high | (high << 4)) & 0x10C3
    low = (low | (low << 4)) & 0x0C30C30C30C3

    # n = (n | (n <<  2)) & 0x 1249 2492 4924 9249
    high = (high | (high << 2)) & 0x1249
    low = (low | (low << 2)) & 0x249249249249

    n = (high << 48) | low
    return n


def unpart1by2(n: Int64[torch.Tensor, "..."]) -> Int32[torch.Tensor, "..."]:
    """
    Inverse of part1by2 - "delete" all bits not at positions divisible by 3
    """
    # mask 21 bits
    n = n & 0x1249249249249249
    n = (n | (n >> 2)) & 0x10C30C30C30C30C3
    n = (n | (n >> 4)) & 0x100F00F00F00F00F
    n = (n | (n >> 8)) & 0x001F0000FF0000FF
    n = (n | (n >> 16)) & 0x001F00000000FFFF
    n = (n | (n >> 32)) & 0x1FFFFF
    return n.to(torch.int32)


def morton_encode(
    indices: Int32[torch.Tensor, "... 3"],
) -> Int64[torch.Tensor, "..."]:
    """Compute Morton code from x, y, z (all torch tensors)"""
    ix = indices[..., 0]
    iy = indices[..., 1]
    iz = indices[..., 2]
    n = 1 << Constants.MAX_OCTREE_DEPTH
    valid = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (iz >= 0) & (iz < n)
    codes = (part1by2(iz) << 2) | (part1by2(iy) << 1) | part1by2(ix)
    codes[~valid] = -1
    return codes


def morton_decode(
    code: Int64[torch.Tensor, "..."],
) -> Int32[torch.Tensor, "... 3"]:
    """Decode Morton code into x, y, z (all torch tensors)"""
    ix = unpart1by2(code >> 0)
    iy = unpart1by2(code >> 1)
    iz = unpart1by2(code >> 2)
    return torch.stack([ix, iy, iz], dim=-1)


def get_ancestor_code(
    code: int, current_octree_depth: int, ancestor_octree_depth: int
) -> int:
    if ancestor_octree_depth < 0:
        raise ValueError(
            "ancestor_octree_depth must be greater than 0",
            f"but {ancestor_octree_depth} < 0",
        )
    if ancestor_octree_depth > current_octree_depth:
        raise ValueError(
            "ancestor_octree_depth must be"
            "less than or equal to current_octree_depth",
            f"but {ancestor_octree_depth} > {current_octree_depth}",
        )
    if ancestor_octree_depth == current_octree_depth:
        return code

    shift = 3 * (Constants.MAX_OCTREE_DEPTH - ancestor_octree_depth)
    mask = ~((1 << shift) - 1)
    return code & mask


def get_parent_code(code: int, octree_depth: int) -> int:
    parent_octree_depth = octree_depth - 1
    return get_ancestor_code(code, octree_depth, parent_octree_depth)


def get_child_codes(code: int, octree_depth: int) -> list[int]:
    if octree_depth >= Constants.MAX_OCTREE_DEPTH:
        raise ValueError(
            "octree_depth must be less than max_octree_depth",
            f"but {octree_depth} >= {Constants.MAX_OCTREE_DEPTH}",
        )
    child_octree_depth = octree_depth + 1
    shift = 3 * (Constants.MAX_OCTREE_DEPTH - child_octree_depth)
    return [code + (i << shift) for i in range(2**3)]


def code_to_local_index(
    code: int, octree_depth: int
) -> Int32[torch.Tensor, " 3"]:
    """Get local index [ix, iy, iz] from Morton code at given depth."""
    _code = torch.tensor(code, dtype=torch.int64)
    shift = 3 * (Constants.MAX_OCTREE_DEPTH - octree_depth)
    shifted = _code >> shift  # to coarse resolution
    return morton_decode(shifted)


def local_index_to_code(
    local_indices: Int32[torch.Tensor, "... 3"], octree_depth: int
) -> Int64[torch.Tensor, "..."]:
    codes = morton_encode(local_indices)
    shift = 3 * (Constants.MAX_OCTREE_DEPTH - octree_depth)
    return codes << shift
