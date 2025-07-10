import dataclasses as dc


@dc.dataclass(init=False, frozen=True)
class Constants:
    """
    Constants for the gridfoam package
    Changing these values breaks the code
    (e.g. the morton code)
    """

    MAX_OCTREE_DEPTH = 20
    MAX_LEVEL = MAX_OCTREE_DEPTH + 1
    MORTON_CODE_BIT_LENGTH = 64
    DEPTH_BIT_LENGTH = 5
    MORTON_ID_BIT_LENGTH = MORTON_CODE_BIT_LENGTH + DEPTH_BIT_LENGTH

