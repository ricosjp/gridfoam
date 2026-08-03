from __future__ import annotations

import fluxel

from gridfoam.core.grid.axis_projected import AxisProjectedGrid
from gridfoam.core.grid.base import IGridBase
from gridfoam.meta.config import GridfoamConfig
from gridfoam.meta.enums import IbmType


def create_grid(config: GridfoamConfig) -> IGridBase:
    """
    Build a computational grid from a validated configuration.

    Parameters
    ----------
    config : GridfoamConfig
        Top-level gridfoam configuration including ``fluxel`` and
        ``simulator`` settings.

    Returns
    -------
    IGridBase
        Constructed grid implementation for the configured IBM type.

    Raises
    ------
    ValueError
        If ``config.fluxel.ibm_type`` is unsupported.
    """
    fluxel_config = config.fluxel
    domain_bbox = fluxel.BoundingBox(
        fluxel_config.domain.lower,
        fluxel_config.domain.upper,
    )
    fluxel_mng = fluxel.FluxelManager(
        domain_bbox,
        fluxel_config.root_resolution,
        n_leaf_refinement=fluxel_config.n_leaf_refinement,
    )
    path = (
        str(fluxel_config.mesh_path)
        if fluxel_config.mesh_path is not None
        else None
    )
    refinement_regions = [
        (region.min, region.max, region.level)
        for region in fluxel_config.refinement_regions
    ]
    match fluxel_config.ibm_type:
        case IbmType.AXIS_PROJECTED:
            fluxel_mesh = fluxel_mng.build_axis_projected_mesh(
                mesh_path=path,
                target_level=fluxel_config.target_level,
                refinement_regions=refinement_regions,
            )
            return AxisProjectedGrid(
                simulator_config=config.simulator,
                fluxel_mesh=fluxel_mesh,
                mesh_path=fluxel_config.mesh_path,
            )
        case _:
            raise ValueError(f"Unsupported IBM type: {config.fluxel.ibm_type}")
