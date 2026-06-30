"""Shared fixtures and helpers for end-to-end force-coefficient tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import torch

from gridfoam.boundaries.basic.dirichlet import DirichletBC
from gridfoam.boundaries.basic.neumann import NeumannBC
from gridfoam.core.field import CellField
from gridfoam.meta.config import (
    BoundaryConditionConfig,
    ConditionConfig,
    ControlConfig,
    DomainConfig,
    FluxelConfig,
    ForceCoeffConfig,
    GridfoamConfig,
    OutputConfig,
    PostProcessingConfig,
    RelaxationFactorsConfig,
    SimulatorConfig,
    SIMPLEAlgorithm,
    SolverConfig,
    fvSchemesConfig,
    fvSolutionConfig,
)
from gridfoam.meta.enums import (
    AlgorithmType,
    BoundaryConditionType,
    DeviceType,
    DivScheme,
    IbmType,
    PrecisionType,
    SolverType,
)
from tests.conftest import default_properties


@pytest.fixture
def make_cube_stl() -> Callable[..., None]:
    """Return a callable that writes an axis-aligned cube STL."""

    def _make(
        *,
        center: tuple[float, float, float],
        side: float,
        out_path: Path,
    ) -> None:
        cx, cy, cz = center
        h = side / 2.0
        box = pv.Box(
            bounds=(cx - h, cx + h, cy - h, cy + h, cz - h, cz + h),
            level=0,
            quads=False,
        )
        box = box.triangulate()
        box.compute_normals(
            point_normals=False, cell_normals=True, inplace=True
        )
        box.save(str(out_path))

    return _make


@pytest.fixture
def make_thin_plate_stl() -> Callable[..., None]:
    """Return a callable that writes a thin rectangular-plate STL."""

    def _make(
        *,
        center: tuple[float, float, float],
        length_x: float,
        length_y: float,
        out_path: Path,
        tilt_deg: float = 0.0,
        thickness: float = 0.02,
    ) -> None:
        cx, cy, cz = center
        hx = length_x / 2.0
        hy = length_y / 2.0
        hz = thickness / 2.0

        verts = np.array(
            [
                [-hx, -hy, -hz],
                [+hx, -hy, -hz],
                [+hx, +hy, -hz],
                [-hx, +hy, -hz],
                [-hx, -hy, +hz],
                [+hx, -hy, +hz],
                [+hx, +hy, +hz],
                [-hx, +hy, +hz],
            ],
            dtype=np.float64,
        )

        theta = np.deg2rad(tilt_deg)
        rot = np.array(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ]
        )
        verts = verts @ rot.T + np.array([cx, cy, cz])

        faces = [
            [3, 0, 2, 1],
            [3, 0, 3, 2],
            [3, 4, 5, 6],
            [3, 4, 6, 7],
            [3, 0, 4, 7],
            [3, 0, 7, 3],
            [3, 1, 2, 6],
            [3, 1, 6, 5],
            [3, 0, 1, 5],
            [3, 0, 5, 4],
            [3, 2, 3, 7],
            [3, 2, 7, 6],
        ]
        face_array = np.concatenate(faces).astype(np.int64)
        mesh = pv.PolyData(verts, face_array)
        mesh = mesh.triangulate()
        mesh.compute_normals(
            point_normals=False, cell_normals=True, inplace=True
        )
        mesh.save(str(out_path))

    return _make


@pytest.fixture
def build_force_config() -> Callable[..., GridfoamConfig]:
    """Return a callable that builds a minimal force-coefficient config."""

    def _build(
        *,
        stl_path: Path,
        root_resolution: list[int],
        output_dir: Path,
        domain_lower: tuple[float, float, float] = (0.0, 0.0, 0.0),
        domain_upper: tuple[float, float, float] = (1.0, 1.0, 1.0),
        drag_dir: tuple[float, float, float] = (1.0, 0.0, 0.0),
        lift_dir: tuple[float, float, float] = (0.0, 0.0, 1.0),
        cor: tuple[float, float, float] = (0.5, 0.5, 0.5),
        A_ref: float = 1.0,
        L_ref: float = 1.0,
        magU_ref: float = 1.0,
        rho: float = 1.0,
        nu: float = 0.1,
    ) -> GridfoamConfig:
        return GridfoamConfig(
            fluxel=FluxelConfig(
                domain=DomainConfig(
                    lower=list(domain_lower),
                    upper=list(domain_upper),
                ),
                root_resolution=root_resolution,
                target_level=0,
                n_leaf_refinement=0,
                mesh_path=stl_path,
                ibm_type=IbmType.AXIS_PROJECTED,
            ),
            simulator=SimulatorConfig(
                control=ControlConfig(
                    deltaT=1.0,
                    endTime=1.0,
                    writeInterval=1,
                    output=OutputConfig(
                        output_dir=output_dir,
                        base_name="force_e2e",
                    ),
                    precision=PrecisionType.FLOAT64,
                ),
                fvSchemes=fvSchemesConfig(),
                fvSolution=fvSolutionConfig(
                    algorithm=SIMPLEAlgorithm(type=AlgorithmType.SIMPLE),
                    solvers={"p": SolverConfig(method=SolverType.CG)},
                ),
                conditions={},
                properties=default_properties(nu=nu),
                post_processing=PostProcessingConfig(
                    forceCoeff=ForceCoeffConfig.model_validate(
                        {
                            "patches": ["_default"],
                            "rho": rho,
                            "magU_ref": magU_ref,
                            "A_ref": A_ref,
                            "L_ref": L_ref,
                            "local_coord": {
                                "drag_dir": list(drag_dir),
                                "lift_dir": list(lift_dir),
                                "center_of_rotation": list(cor),
                            },
                        }
                    ),
                ),
                device=DeviceType.CPU,
            ),
        )

    return _build


@pytest.fixture
def build_cube_simple_config() -> Callable[..., GridfoamConfig]:
    """Return a callable that builds a full SIMPLE-ready cube config."""

    def _build(
        *,
        stl_path: Path,
        output_dir: Path,
        root_resolution: list[int],
        domain_lower: tuple[float, float, float],
        domain_upper: tuple[float, float, float],
        cor: tuple[float, float, float],
        A_ref: float,
        L_ref: float,
        end_time: float = 20.0,
        nu: float = 0.05,
    ) -> GridfoamConfig:
        return GridfoamConfig(
            fluxel=FluxelConfig(
                domain=DomainConfig(
                    lower=list(domain_lower),
                    upper=list(domain_upper),
                ),
                root_resolution=root_resolution,
                target_level=0,
                n_leaf_refinement=0,
                mesh_path=stl_path,
                ibm_type=IbmType.AXIS_PROJECTED,
            ),
            simulator=SimulatorConfig(
                control=ControlConfig(
                    deltaT=1.0,
                    endTime=end_time,
                    writeInterval=int(end_time),
                    output=OutputConfig(
                        output_dir=output_dir,
                        base_name="cube_e2e",
                    ),
                    precision=PrecisionType.FLOAT64,
                ),
                fvSchemes=fvSchemesConfig(
                    divSchemes={"default": DivScheme.UPWIND},
                ),
                fvSolution=fvSolutionConfig(
                    algorithm=SIMPLEAlgorithm(
                        type=AlgorithmType.SIMPLE,
                        relaxationFactors=RelaxationFactorsConfig(
                            equations={"U": 0.7, "p": 0.3},
                        ),
                        pRefCell=0,
                        pRefValue=0.0,
                    ),
                    solvers={
                        "U": SolverConfig(
                            method=SolverType.BiCGSTAB,
                            max_iter=500,
                            rel_tolerance=0.1,
                            tolerance=1e-8,
                        ),
                        "p": SolverConfig(
                            method=SolverType.CG,
                            max_iter=500,
                            rel_tolerance=0.01,
                            tolerance=1e-7,
                        ),
                    },
                ),
                conditions={
                    "U": ConditionConfig(
                        internal=[0.0, 0.0, 0.0],
                        boundary={
                            "inlet": BoundaryConditionConfig(
                                type=BoundaryConditionType.DIRICHLET,
                                patches=["x_minus"],
                                value=[1.0, 0.0, 0.0],
                            ),
                            "outlet": BoundaryConditionConfig(
                                type=BoundaryConditionType.INLET_OUTLET,
                                patches=["x_plus"],
                                value=[0.0, 0.0, 0.0],
                            ),
                            "slip": BoundaryConditionConfig(
                                type=BoundaryConditionType.SLIP,
                                patches=[
                                    "y_minus",
                                    "y_plus",
                                    "z_minus",
                                    "z_plus",
                                ],
                            ),
                            "body": BoundaryConditionConfig(
                                type=BoundaryConditionType.DIRICHLET,
                                patches=["_default"],
                                value=[0.0, 0.0, 0.0],
                            ),
                        },
                    ),
                    "p": ConditionConfig(
                        internal=[0.0],
                        boundary={
                            "inlet": BoundaryConditionConfig(
                                type=BoundaryConditionType.NEUMANN,
                                patches=["x_minus"],
                                value=[0.0],
                            ),
                            "outlet": BoundaryConditionConfig(
                                type=BoundaryConditionType.DIRICHLET,
                                patches=["x_plus"],
                                value=[0.0],
                            ),
                            "slip": BoundaryConditionConfig(
                                type=BoundaryConditionType.NEUMANN,
                                patches=[
                                    "y_minus",
                                    "y_plus",
                                    "z_minus",
                                    "z_plus",
                                ],
                                value=[0.0],
                            ),
                            "body": BoundaryConditionConfig(
                                type=BoundaryConditionType.NEUMANN,
                                patches=["_default"],
                                value=[0.0],
                            ),
                        },
                    ),
                },
                properties=default_properties(nu=nu),
                post_processing=PostProcessingConfig(
                    forceCoeff=ForceCoeffConfig.model_validate(
                        {
                            "patches": ["_default"],
                            "rho": 1.0,
                            "magU_ref": 1.0,
                            "A_ref": A_ref,
                            "L_ref": L_ref,
                            "local_coord": {
                                "drag_dir": [1.0, 0.0, 0.0],
                                "lift_dir": [0.0, 0.0, 1.0],
                                "center_of_rotation": list(cor),
                            },
                        }
                    ),
                ),
                device=DeviceType.CPU,
            ),
        )

    return _build


@pytest.fixture
def add_dirichlet_bc() -> Callable[..., None]:
    """Return a callable that attaches a Dirichlet boundary condition."""

    def _add(field: CellField, patch: str, value: list[float]) -> None:
        bc = DirichletBC(
            torch.tensor(
                value, dtype=field.grid.dtype, device=field.grid.device
            )
        )
        field.add_boundary_conditions({patch: bc})

    return _add


@pytest.fixture
def add_neumann_bc() -> Callable[..., None]:
    """Return a callable that attaches a Neumann boundary condition."""

    def _add(field: CellField, patch: str, value: list[float]) -> None:
        bc = NeumannBC(
            torch.tensor(
                value, dtype=field.grid.dtype, device=field.grid.device
            )
        )
        field.add_boundary_conditions({patch: bc})

    return _add
