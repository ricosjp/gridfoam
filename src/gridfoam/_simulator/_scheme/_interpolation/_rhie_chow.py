from gridfoam._base import TensorGrid, iter_leaf_cubes_of


class RhieChowInterpolation:
    """
    Rhie-Chow interpolation for face-centered velocity computation.

    This class implements the Rhie-Chow interpolation scheme for computing
    face-centered velocities from cell-centered velocities. The Rhie-Chow
    interpolation helps to prevent pressure-velocity decoupling in collocated
    grid arrangements.
    """

    def __init__(self, velocity_name: str):
        """
        Initialize the Rhie-Chow interpolation.

        Parameters
        ----------
        velocity_name : str
            Name of the cell-centered velocity field.
        """
        self.U_i_name = velocity_name
        self.U_f_name = "_" + velocity_name + "_f"

    def run(self, grid: TensorGrid) -> None:
        """
        Run the Rhie-Chow interpolation on the grid.

        This method computes face-centered velocities from cell-centered
        velocities for all leaf cubes in the grid. Currently uses simple
        face averaging, with Rhie-Chow correction to be implemented.

        Parameters
        ----------
        grid : TensorGrid
            The tensor grid containing the velocity fields.
        """
        for octree_level in grid.data.octree_levels:
            for cube in iter_leaf_cubes_of(octree_level):
                U_i = cube.old.cells[self.U_i_name]
                U_f = U_i.face_average()  # TODO: Add Rhie-Chow correction
                cube.old.faces[self.U_f_name] = U_f
                cube.cur.faces[self.U_f_name] = U_f
