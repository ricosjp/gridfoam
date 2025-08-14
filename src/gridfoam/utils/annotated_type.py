from typing import Annotated

CubeCode = Annotated[
    int, "Unique cube id: (root_code, morton_code) -> int"
]


NeighborCodeList = Annotated[list[CubeCode|None], "length=26"]
