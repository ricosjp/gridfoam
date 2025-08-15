from typing import Annotated

CubeCode = Annotated[int, "(root_code, morton_code)"]
NeighborCodeList = Annotated[list[CubeCode | None], "length=26"]
