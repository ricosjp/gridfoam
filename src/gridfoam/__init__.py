# import nanofoam as m

# def main() -> None:
#     print("Hello from gridfoam!")
#     print(m.add(1, 2))
#     print(m.subtract(1, 2))
#     print(m.multiply(2, 3))
#     print(m.divide(6, 3))

# main()

from importlib.metadata import version

from gridfoam.tensor import GridTensor

__version__ = version("gridfoam")
__all__ = ["GridTensor"]
