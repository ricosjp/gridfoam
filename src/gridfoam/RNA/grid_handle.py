import pathlib
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Float, Int

from gridfoam.DNA._grid._grid import (
    NodeType,
    PyOctreeLevel,
    PyOctreeNode,
    RawIndexConversionMode,
    generate_grid_from_polydata,
)
from gridfoam.DNA._grid._grid import PyGrid as Grid
from gridfoam.DNA._gridhandle import IGridHandle
from gridfoam.DNA.ASTNodes._interface import IASTNode
from gridfoam.DNA.ASTNodes.arithmetic_node import ArithmeticNode, ArithmeticType
from gridfoam.DNA.ASTNodes.operator_node import OperatorNode
from gridfoam.DNA.config import GridfoamConfig, YamlRoot
from gridfoam.DNA.constants import FACE_NEIGHBOR_MAP
from gridfoam.DNA.ctx_for_cube_operation import CtxForCubeOperation
from gridfoam.DNA.cubefield import CubeField
from gridfoam.DNA.enum import FieldLayout
from gridfoam.DNA.fielddata import FVMatrix
from gridfoam.DNA.meta.equation import EquationMeta
from gridfoam.DNA.meta.field import FieldMeta
from gridfoam.DNA.scheme.fvm.identity import FVMIdentity
from gridfoam.RNA.registry import SimulationMetaRegistry


def _generate_grid_indices(
    divisions: Int[torch.Tensor, " 3"],
) -> Int[torch.Tensor, "3 n_grid"]:
    """
    Generate 3D grid indices
    according to the number of divisions along each axis.

    The output indices are ordered in Z-order.

    Parameters
    ----------
    divisions : Int32[torch.Tensor, " 3"]
        The number of divisions along each axis (X, Y, Z).

    Returns
    -------
    Int32[torch.Tensor, "3 n_grid"]
        Tensor of shape (3, n_grid) containing all grid indices,
        where n_grid = divisions[0] * divisions[1] * divisions[2].
        Each row corresponds:
        - 0: x-axis
        - 1: y-axis
        - 2: z-axis
    """
    indices_per_axis = [
        torch.arange(n, dtype=divisions.dtype, device=divisions.device)
        for n in reversed(divisions)
    ]
    indices = torch.meshgrid(*indices_per_axis, indexing="ij")
    indices = torch.stack(indices[::-1], dim=0).reshape(3, -1)
    return indices


class GridHandle(IGridHandle):
    def __init__(self, configpath: pathlib.Path) -> None:
        with open(configpath) as f:
            raw_yaml = yaml.safe_load(f)
        self._config = YamlRoot.model_validate(raw_yaml).gridfoam
        meshpath = self._config.mesh.file
        self._mesh = pv.read(meshpath)

        if not isinstance(self._mesh, pv.PolyData):
            raise ValueError("Mesh must be a PolyData")
        if not self._mesh.is_all_triangles:
            self._mesh = self._mesh.triangulate()

        self._grid = generate_grid_from_polydata(self._mesh, configpath)

    ## Iterating over the grid
    def iter_levels(self) -> Iterator[PyOctreeLevel]:
        return iter(self._grid.octree_levels)

    def iter_leaf_on_level(
        self, level: PyOctreeLevel
    ) -> Iterator[PyOctreeNode]:
        for cube in level.nodes.values():
            if cube.node_type == NodeType.LEAF:
                yield cube

    def iter_gfp_on_level(self, level: PyOctreeLevel) -> Iterator[PyOctreeNode]:
        for cube in level.nodes.values():
            if cube.node_type == NodeType.GHOST_FROM_PARENT:
                yield cube

    def iter_gfc_on_level(self, level: PyOctreeLevel) -> Iterator[PyOctreeNode]:
        for cube in level.nodes.values():
            if cube.node_type == NodeType.GHOST_FROM_CHILD:
                yield cube

    def iter_all_leaves(self) -> Iterator[tuple[int, PyOctreeNode]]:
        for level in self.iter_levels():
            for cube in self.iter_leaf_on_level(level):
                yield level.depth, cube

    ## Allocating fields
    def _calculate_pos(
        self, depth: int, cube: PyOctreeNode
    ) -> tuple[
        Float[torch.Tensor, "N N N"],
        Float[torch.Tensor, "N N N"],
        Float[torch.Tensor, "N N N"],
    ]:
        N = self._config.cube.interior_width
        H = self._config.cube.halo_width
        W = N + 2 * H
        device = self._config.cube.device
        divisions = torch.tensor([W, W, W], dtype=torch.int64, device=device)
        dx = self.get_dx_at_depth(depth)  # (3,)
        cube_index = torch.tensor(
            cube.cubecode.to_global_index(depth).astype(np.int64),
            dtype=torch.int64,
            device=cube.field.device,
        )  # (3,)
        domain_lower_pos = torch.tensor(
            self._grid.domain.lower,
            device=cube.field.device,
        )  # (3,)
        cell_index_in_cube = _generate_grid_indices(divisions).reshape(
            3, W, W, W
        )  # (3, W, W, W)
        cell_index_in_global = (
            cell_index_in_cube - H + cube_index[:, None, None, None] * N
        )  # (3, W, W, W)
        x, y, z = (
            domain_lower_pos[:, None, None, None]
            + (2 * cell_index_in_global + 1) * 0.5 * dx[:, None, None, None]
        )
        return x, y, z

    def allocate_by_registry(self, registry: SimulationMetaRegistry) -> None:
        for level in self.iter_levels():
            dx = self.get_dx_at_depth(level.depth)
            for cube in level.nodes.values():
                cube.field = CubeField(self._config.cube)
                x, y, z = self._calculate_pos(level.depth, cube)
                for field_meta in registry.fields.values():
                    cube.field.add_field(field_meta)
                    if field_meta.initialize_func is not None:
                        cell_field = cube.field.get_field(field_meta)
                        cell_field.raw[0] = field_meta.initialize_func(
                            x, y, z
                        )
                    if field_meta.name == "phi":
                        U = cube.field.get_field(registry.get_field("U"))
                        phi = cube.field.get_field(field_meta)
                        Uf = U.face_average()
                        phi.x[0] = Uf.x[0,0] * dx[1] * dx[2]
                        phi.y[0] = Uf.y[0,1] * dx[0] * dx[2]
                        phi.z[0] = Uf.z[0,2] * dx[0] * dx[1]
                for equation_meta in registry.equations.values():
                    cube.field.add_equation(equation_meta)
        sync_list = [
            fm
            for fm in registry.fields.values()
            if fm.layout == FieldLayout.CELL
        ]
        self.sync_all(sync_list)

    def allocate_field(self, field_meta: FieldMeta) -> None:
        for level in self.iter_levels():
            for cube in level.nodes.values():
                if cube.field is None:
                    cube.field = CubeField(self._config.cube)
                cube.field.add_field(field_meta)

    def allocate_equation(self, equation_meta: EquationMeta) -> None:
        for level in self.iter_levels():
            for cube in level.nodes.values():
                if cube.field is None:
                    cube.field = CubeField(self._config.cube)
                cube.field.add_equation(equation_meta)

    def update_fvmatrix(self, equation_meta: EquationMeta) -> None:
        for level in self.iter_levels():
            dx = self.get_dx_at_depth(level.depth)
            for cube in level.nodes.values():
                ctx = CtxForCubeOperation(
                    depth=level.depth,
                    bounds=level.bounds,
                    dt=self._config.simulator.control.deltaT,
                    dx=dx,
                    vertices=torch.tensor(
                        self._mesh.points,
                        dtype=torch.float32,
                        device=self._config.cube.device,
                    ),
                    bcs=equation_meta.boundary_conditions,
                )
                fvmatrix = self._evaluate_node(
                    equation_meta.ast_root, cube, ctx
                )
                cube.field.fvmatrices[equation_meta.name] = fvmatrix
        self.sync_all(em_list=[equation_meta])

    def _evaluate_node(
        self, node: IASTNode, cube: PyOctreeNode, ctx: CtxForCubeOperation
    ) -> FVMatrix:
        if isinstance(node, ArithmeticNode):
            match node.type:
                case ArithmeticType.ADD:
                    return self._evaluate_node(
                        node.arg1, cube, ctx
                    ) + self._evaluate_node(node.arg2, cube, ctx)
                case ArithmeticType.SUB:
                    return self._evaluate_node(
                        node.arg1, cube, ctx
                    ) - self._evaluate_node(node.arg2, cube, ctx)
                case _:
                    raise ValueError(f"Unknown arithmetic type: {node.type}")
        if isinstance(node, OperatorNode):
            return node.operator.build(cube, ctx)
        if isinstance(node, FieldMeta):
            return FVMIdentity(node).build(cube, ctx)

    ## Synchronizing halo
    def sync_halo_at_depth(
        self,
        depth: int,
        fm_list: list[FieldMeta] | None = None,
        em_list: list[EquationMeta] | None = None,
    ) -> None:
        """
        Synchronize the halo for a given depth.

        Parameters
        ----------
        depth : int
            Depth to synchronize the halo at.
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        em_list : list[EquationMeta]
            List of equation metas to synchronize.
        """
        if fm_list is None:
            fm_list = []
        if em_list is None:
            em_list = []
        cell_fm_list = [fm for fm in fm_list if fm.layout == FieldLayout.CELL]
        # face_fm_list = [fm for fm in fm_list if fm.layout == FieldLayout.FACE]

        level = self._grid.octree_levels[depth]
        bounds = level.bounds
        for cube in self.iter_leaf_on_level(level):
            nbr_codes = cube.cubecode.neighbor_codes(
                depth,
                bounds,
                RawIndexConversionMode.BORDER,
                False,
            )
            # iterate over the face neighbors
            for f, (axis, forward) in FACE_NEIGHBOR_MAP.items():
                nbr_code = nbr_codes[f]
                if nbr_code is None:
                    continue
                nbr_cube = level.nodes[nbr_code.value()]
                for fm in cell_fm_list:
                    tgt_cell_field = cube.field.cells[fm.name]
                    nbr_cell_field = nbr_cube.field.cells[fm.name]
                    nbr_interior_halo = nbr_cell_field.get_interior_halo_along(
                        axis, not forward
                    )
                    tgt_cell_field.set_halo_along(
                        axis, forward, nbr_interior_halo
                    )
                for em in em_list:
                    tgt_fvmatrix = cube.field.fvmatrices[em.name]
                    nbr_fvmatrix = nbr_cube.field.fvmatrices[em.name]
                    # a_P
                    nbr_a_P_interior_halo = (
                        nbr_fvmatrix.a_P.get_interior_halo_along(
                            axis, not forward
                        )
                    )
                    tgt_fvmatrix.a_P.set_halo_along(
                        axis, forward, nbr_a_P_interior_halo
                    )
                    # a_E
                    nbr_a_E_interior_halo = (
                        nbr_fvmatrix.a_E.get_interior_halo_along(
                            axis, not forward
                        )
                    )
                    tgt_fvmatrix.a_E.set_halo_along(
                        axis, forward, nbr_a_E_interior_halo
                    )
                    # a_W
                    nbr_a_W_interior_halo = (
                        nbr_fvmatrix.a_W.get_interior_halo_along(
                            axis, not forward
                        )
                    )
                    tgt_fvmatrix.a_W.set_halo_along(
                        axis, forward, nbr_a_W_interior_halo
                    )
                    # a_N
                    nbr_a_N_interior_halo = (
                        nbr_fvmatrix.a_N.get_interior_halo_along(
                            axis, not forward
                        )
                    )
                    tgt_fvmatrix.a_N.set_halo_along(
                        axis, forward, nbr_a_N_interior_halo
                    )
                    # a_S
                    nbr_a_S_interior_halo = (
                        nbr_fvmatrix.a_S.get_interior_halo_along(
                            axis, not forward
                        )
                    )
                    tgt_fvmatrix.a_S.set_halo_along(
                        axis, forward, nbr_a_S_interior_halo
                    )
                    # a_T
                    nbr_a_T_interior_halo = (
                        nbr_fvmatrix.a_T.get_interior_halo_along(
                            axis, not forward
                        )
                    )
                    tgt_fvmatrix.a_T.set_halo_along(
                        axis, forward, nbr_a_T_interior_halo
                    )
                    # a_B
                    nbr_a_B_interior_halo = (
                        nbr_fvmatrix.a_B.get_interior_halo_along(
                            axis, not forward
                        )
                    )
                    tgt_fvmatrix.a_B.set_halo_along(
                        axis, forward, nbr_a_B_interior_halo
                    )
                    # source
                    nbr_source_interior_halo = (
                        nbr_fvmatrix.source.get_interior_halo_along(
                            axis, not forward
                        )
                    )
                    tgt_fvmatrix.source.set_halo_along(
                        axis, forward, nbr_source_interior_halo
                    )

    def sync_halo(
        self,
        fm_list: list[FieldMeta] | None = None,
        em_list: list[EquationMeta] | None = None,
    ) -> None:
        """
        Synchronize the halo for all depths.

        Parameters
        ----------
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        em_list : list[EquationMeta]
            List of equation metas to synchronize.
        """
        if fm_list is None:
            fm_list = []
        if em_list is None:
            em_list = []
        for depth in range(self._grid.max_depth):
            self.sync_halo_at_depth(depth, fm_list, em_list)

    ## Synchronizing ghost from parent
    def sync_gfp_at_depth(
        self,
        depth: int,
        fm_list: list[FieldMeta] | None = None,
        em_list: list[EquationMeta] | None = None,
    ) -> None:
        """
        Synchronize ghost cubes from their parent cubes at a given depth.

        This method extracts data from parent cubes and distributes it to
        ghost cubes that are refined versions of their parent cubes.
        The data is interpolated using trilinear interpolation.

        Parameters
        ----------
        depth : int
            The depth level to synchronize.
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        em_list : list[EquationMeta]
            List of equation metas to synchronize.
        """
        if fm_list is None:
            fm_list = []
        if em_list is None:
            em_list = []
        cell_fm_list = [fm for fm in fm_list if fm.layout == FieldLayout.CELL]
        # face_fm_list = [fm for fm in fm_list if fm.layout == FieldLayout.FACE]

        level = self._grid.octree_levels[depth]
        for cube in self.iter_gfp_on_level(level):
            # Get parent cube code and offsets within parent
            parent_code, offsets = cube.cubecode.parent_and_offset_py(depth)
            parent_cube = self._grid.octree_levels[depth - 1].nodes[
                parent_code.value()
            ]

            # TODO:
            # 1. Piecewise linear reconstruction
            # https://zingale.github.io/comp_astro_tutorial/advection_euler/advection/advection-partIII.html
            # 2. Piecewise Parabolic Method

            for fm in cell_fm_list:
                tgt_cell_tensor = cube.field.cells[fm.name]
                parent_cell_tensor = parent_cube.field.cells[fm.name]
                parent_half_interior = parent_cell_tensor.get_half_interior(
                    offsets
                )  # (T C halfN halfN halfN)

                # interpolation
                refined_tensor: torch.Tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)

                # assign the refined data to the ghost cube
                tgt_cell_tensor.interior = refined_tensor
            for em in em_list:
                tgt_fvmatrix = cube.field.fvmatrices[em.name]
                parent_fvmatrix = parent_cube.field.fvmatrices[em.name]
                # a_P
                parent_half_interior = parent_fvmatrix.a_P.get_half_interior(
                    offsets
                )
                refined_tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)
                tgt_fvmatrix.a_P.interior = refined_tensor
                # a_E
                parent_half_interior = parent_fvmatrix.a_E.get_half_interior(
                    offsets
                )
                refined_tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)
                tgt_fvmatrix.a_E.interior = refined_tensor
                # a_W
                parent_half_interior = parent_fvmatrix.a_W.get_half_interior(
                    offsets
                )
                refined_tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)
                tgt_fvmatrix.a_W.interior = refined_tensor
                # a_N
                parent_half_interior = parent_fvmatrix.a_N.get_half_interior(
                    offsets
                )
                refined_tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)
                tgt_fvmatrix.a_N.interior = refined_tensor
                # a_S
                parent_half_interior = parent_fvmatrix.a_S.get_half_interior(
                    offsets
                )
                refined_tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)
                tgt_fvmatrix.a_S.interior = refined_tensor
                # a_T
                parent_half_interior = parent_fvmatrix.a_T.get_half_interior(
                    offsets
                )
                refined_tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)
                tgt_fvmatrix.a_T.interior = refined_tensor
                # a_B
                parent_half_interior = parent_fvmatrix.a_B.get_half_interior(
                    offsets
                )
                refined_tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)
                tgt_fvmatrix.a_B.interior = refined_tensor
                # source
                parent_half_interior = parent_fvmatrix.source.get_half_interior(
                    offsets
                )
                refined_tensor = F.interpolate(
                    parent_half_interior,
                    scale_factor=2,
                    mode="trilinear",
                )  # (T C N N N)
                tgt_fvmatrix.source.interior = refined_tensor

    def sync_gfp(
        self,
        fm_list: list[FieldMeta] | None = None,
        em_list: list[EquationMeta] | None = None,
    ) -> None:
        """
        Synchronize ghost cubes from their parent cubes for all depths.

        This method extracts data from parent cubes and distributes it to
        ghost cubes that are refined versions of their parent cubes.
        The data is interpolated using trilinear interpolation.

        Parameters
        ----------
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        em_list : list[EquationMeta]
            List of equation metas to synchronize.
        """
        if fm_list is None:
            fm_list = []
        if em_list is None:
            em_list = []
        for depth in range(self._grid.max_depth):
            if depth == 0:
                continue
            self.sync_gfp_at_depth(depth, fm_list, em_list)

    ## Synchronizing ghost from child
    def sync_gfc_at_depth(
        self,
        depth: int,
        fm_list: list[FieldMeta] | None = None,
        em_list: list[EquationMeta] | None = None,
    ) -> None:
        """
        Synchronize ghost cubes from their child cubes at a given depth.

        This method aggregates data from child cubes and distributes it to
        ghost cubes that are coarsened versions of their child cubes.
        The data is coarsened using average pooling
        (2x2x2 cells are averaged into 1 cell).

        Parameters
        ----------
        depth : int
            The depth level to synchronize.
        fm_list : list[FieldMeta]
            List of field metas to synchronize.
        em_list : list[EquationMeta]
            List of equation metas to synchronize.
        """
        if fm_list is None:
            fm_list = []
        if em_list is None:
            em_list = []
        cell_fm_list = [fm for fm in fm_list if fm.layout == FieldLayout.CELL]
        # face_fm_list = [fm for fm in fm_list if fm.layout == FieldLayout.FACE]

        level = self._grid.octree_levels[depth]
        for cube in self.iter_gfc_on_level(level):
            child_codes = cube.cubecode.children(depth)
            # key: field name, value: list of coarsened tensors
            coarsened_fm_tensors = defaultdict(list)
            # key: equation name, value: list of coarsened tensors
            coarsened_aP_tensors = defaultdict(list)
            coarsened_aE_tensors = defaultdict(list)
            coarsened_aW_tensors = defaultdict(list)
            coarsened_aN_tensors = defaultdict(list)
            coarsened_aS_tensors = defaultdict(list)
            coarsened_aT_tensors = defaultdict(list)
            coarsened_aB_tensors = defaultdict(list)
            coarsened_source_tensors = defaultdict(list)
            for child_code in child_codes:
                child_cube = self._grid.octree_levels[depth + 1].nodes[
                    child_code.value()
                ]
                for fm in cell_fm_list:
                    child_interior = child_cube.field.cells[
                        fm.name
                    ].interior  # (T C N N N)

                    # interpolation
                    coarsened_tensor = F.avg_pool3d(
                        child_interior,
                        kernel_size=2,
                        stride=2,
                    )  # (T C halfN halfN halfN)

                    coarsened_fm_tensors[fm.name].append(coarsened_tensor)
                for em in em_list:
                    child_fvmatrix = child_cube.field.fvmatrices[em.name]
                    child_aP_tensor = child_fvmatrix.a_P.interior
                    child_aE_tensor = child_fvmatrix.a_E.interior
                    child_aW_tensor = child_fvmatrix.a_W.interior
                    child_aN_tensor = child_fvmatrix.a_N.interior
                    child_aS_tensor = child_fvmatrix.a_S.interior
                    child_aT_tensor = child_fvmatrix.a_T.interior
                    child_aB_tensor = child_fvmatrix.a_B.interior
                    child_source_tensor = child_fvmatrix.source.interior

                    coarsened_aP_tensor = F.avg_pool3d(
                        child_aP_tensor,
                        kernel_size=2,
                        stride=2,
                    )
                    coarsened_aE_tensor = F.avg_pool3d(
                        child_aE_tensor,
                        kernel_size=2,
                        stride=2,
                    )
                    coarsened_aW_tensor = F.avg_pool3d(
                        child_aW_tensor,
                        kernel_size=2,
                        stride=2,
                    )
                    coarsened_aN_tensor = F.avg_pool3d(
                        child_aN_tensor,
                        kernel_size=2,
                        stride=2,
                    )
                    coarsened_aS_tensor = F.avg_pool3d(
                        child_aS_tensor,
                        kernel_size=2,
                        stride=2,
                    )
                    coarsened_aT_tensor = F.avg_pool3d(
                        child_aT_tensor,
                        kernel_size=2,
                        stride=2,
                    )
                    coarsened_aB_tensor = F.avg_pool3d(
                        child_aB_tensor,
                        kernel_size=2,
                        stride=2,
                    )
                    coarsened_source_tensor = F.avg_pool3d(
                        child_source_tensor,
                        kernel_size=2,
                        stride=2,
                    )
                    coarsened_aP_tensors[em.name].append(coarsened_aP_tensor)
                    coarsened_aE_tensors[em.name].append(coarsened_aE_tensor)
                    coarsened_aW_tensors[em.name].append(coarsened_aW_tensor)
                    coarsened_aN_tensors[em.name].append(coarsened_aN_tensor)
                    coarsened_aS_tensors[em.name].append(coarsened_aS_tensor)
                    coarsened_aT_tensors[em.name].append(coarsened_aT_tensor)
                    coarsened_aB_tensors[em.name].append(coarsened_aB_tensor)
                    coarsened_source_tensors[em.name].append(
                        coarsened_source_tensor
                    )

            # fm
            for fm in cell_fm_list:
                tgt_cell_field = cube.field.cells[fm.name]
                half_width = tgt_cell_field.N // 2  # (T C halfN halfN halfN)
                coarsened_tensors = coarsened_fm_tensors[fm.name]
                for i, tensor in enumerate(coarsened_tensors):
                    iz = i // (2 * 2)
                    iy = (i % (2 * 2)) // 2
                    ix = i % 2
                    xs, xe = ix * half_width, (ix + 1) * half_width
                    ys, ye = iy * half_width, (iy + 1) * half_width
                    zs, ze = iz * half_width, (iz + 1) * half_width
                    tgt_cell_field.interior[..., zs:ze, ys:ye, xs:xe] = tensor

            # em
            for em in em_list:
                tgt_fvmatrix = cube.field.fvmatrices[em.name]
                half_width = tgt_fvmatrix.N // 2  # (T C halfN halfN halfN)
                coarsened_aP_tensors = coarsened_aP_tensors[em.name]
                coarsened_aE_tensors = coarsened_aE_tensors[em.name]
                coarsened_aW_tensors = coarsened_aW_tensors[em.name]
                coarsened_aN_tensors = coarsened_aN_tensors[em.name]
                coarsened_aS_tensors = coarsened_aS_tensors[em.name]
                coarsened_aT_tensors = coarsened_aT_tensors[em.name]
                coarsened_aB_tensors = coarsened_aB_tensors[em.name]
                coarsened_source_tensors = coarsened_source_tensors[em.name]
                coeff_zip = zip(
                    coarsened_aP_tensors,
                    coarsened_aE_tensors,
                    coarsened_aW_tensors,
                    coarsened_aN_tensors,
                    coarsened_aS_tensors,
                    coarsened_aT_tensors,
                    coarsened_aB_tensors,
                    coarsened_source_tensors,
                    strict=True,
                )
                for i, (aP, aE, aW, aN, aS, aT, aB, source) in enumerate(
                    coeff_zip
                ):
                    iz = i // (2 * 2)
                    iy = (i % (2 * 2)) // 2
                    ix = i % 2
                    xs, xe = ix * half_width, (ix + 1) * half_width
                    ys, ye = iy * half_width, (iy + 1) * half_width
                    zs, ze = iz * half_width, (iz + 1) * half_width
                    tgt_fvmatrix.a_P.interior[..., zs:ze, ys:ye, xs:xe] = aP
                    tgt_fvmatrix.a_E.interior[..., zs:ze, ys:ye, xs:xe] = aE
                    tgt_fvmatrix.a_W.interior[..., zs:ze, ys:ye, xs:xe] = aW
                    tgt_fvmatrix.a_N.interior[..., zs:ze, ys:ye, xs:xe] = aN
                    tgt_fvmatrix.a_S.interior[..., zs:ze, ys:ye, xs:xe] = aS
                    tgt_fvmatrix.a_T.interior[..., zs:ze, ys:ye, xs:xe] = aT
                    tgt_fvmatrix.a_B.interior[..., zs:ze, ys:ye, xs:xe] = aB
                    tgt_fvmatrix.source.interior[..., zs:ze, ys:ye, xs:xe] = (
                        source
                    )

    def sync_gfc(
        self,
        fm_list: list[FieldMeta] | None = None,
        em_list: list[EquationMeta] | None = None,
    ) -> None:
        """
        Synchronize ghost cubes from their child cubes for all depths.

        This method aggregates data from child cubes
        and distributes it to ghost cubes
        that are coarsened versions of their child cubes.
        The data is coarsened using average pooling
        (2x2x2 cells are averaged into 1 cell).
        """
        if fm_list is None:
            fm_list = []
        if em_list is None:
            em_list = []
        for depth in reversed(range(self._grid.max_depth)):
            if depth == self._grid.max_depth - 1:
                continue
            self.sync_gfc_at_depth(depth, fm_list, em_list)

    def sync_all(
        self,
        fm_list: list[FieldMeta] | None = None,
        em_list: list[EquationMeta] | None = None,
    ) -> None:
        """
        Synchronize all fields for all depths.
        """
        if fm_list is None:
            fm_list = []
        if em_list is None:
            em_list = []
        self.sync_gfp(fm_list, em_list)
        self.sync_gfc(fm_list, em_list)
        self.sync_halo(fm_list, em_list)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def mesh(self) -> pv.PolyData:
        return self._mesh

    @property
    def config(self) -> GridfoamConfig:
        return self._config

    def get_dx_at_depth(self, depth: int) -> Float[torch.Tensor, " 3"]:
        level = self._grid.octree_levels[depth]
        bounds = level.bounds * self._config.cube.interior_width
        dx = (self._grid.domain.upper - self._grid.domain.lower) / bounds
        return torch.tensor(dx)
