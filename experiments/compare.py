import numpy as np
import pyvista as pv

from experiments.schema import CompareConfig


def compare_on_slice(
    of_mesh: pv.DataSet,
    gf_mesh: pv.DataSet,
    cfg: CompareConfig,
) -> pv.DataSet:
    # use gridfoam slice as base for comparison
    slc = of_mesh.slice(
        normal=cfg.plot.get_slice_normal(),
        origin=cfg.plot.slice_origin,
        generate_triangles=True,
    )
    slc.clear_data()
    assert isinstance(slc, pv.DataSet)

    of_sampled = slc.cell_centers().sample(
        of_mesh, pass_cell_data=True, snap_to_closest_point=True
    )
    gf_sampled = slc.cell_centers().sample(
        gf_mesh, pass_cell_data=True, snap_to_closest_point=True
    )

    of_vals = of_sampled.point_data[cfg.field_name]
    gf_vals = gf_sampled.point_data[cfg.field_name]

    match cfg.field_kind:
        case "scalar":
            of_vals = of_vals.ravel()
            gf_vals = gf_vals.ravel()
            scalar_name = cfg.field_name
        case "vector":
            of_vals = np.linalg.vector_norm(of_vals, axis=1)
            gf_vals = np.linalg.vector_norm(gf_vals, axis=1)
            scalar_name = "Mag " + cfg.field_name
        case _:
            raise ValueError(f"Invalid field kind: {cfg.field_kind}")

    err = of_vals - gf_vals
    slc.cell_data["openfoam"] = of_vals
    slc.cell_data["gridfoam"] = gf_vals
    slc.cell_data["error_abs"] = np.abs(err)

    # metadata
    slc.field_data["scalar_name"] = scalar_name

    # metrics
    slc.field_data["Linf"] = np.max(np.abs(err))
    slc.field_data["L2"] = np.sqrt(np.sum(err**2))
    slc.field_data["L1"] = np.mean(np.abs(err))

    return slc
