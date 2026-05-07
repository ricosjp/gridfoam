import pyvista as pv

from experiments.schema import CompareConfig


def load_openfoam_mesh(cfg: CompareConfig) -> pv.DataSet:
    vtk_dir = cfg.case_dir / "of" / "VTK"
    vtu = next(vtk_dir.rglob("internal.vtu"), None)
    if vtu is None:
        msg = f"No internal.vtu found in {vtk_dir}"
        raise FileNotFoundError(msg)
    return pv.read(vtu)


def load_gridfoam_mesh(cfg: CompareConfig) -> pv.DataSet:
    vtu_dir = cfg.case_dir / "gridfoam" / "outputs"
    vtus = vtu_dir.glob("*.vtu")
    if not vtus:
        msg = f"No .vtu files found in {vtu_dir}"
        raise FileNotFoundError(msg)
    vtu = sorted(vtus)[-1]
    return pv.read(vtu)
