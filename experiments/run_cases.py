import subprocess

from experiments.schema import CompareConfig

of_bashrc = "/usr/lib/openfoam/openfoam2412/etc/bashrc"


def run_openfoam(cfg: CompareConfig) -> None:
    if not cfg.openfoam.enabled:
        return
    of_case = cfg.case_dir / "of"
    if not of_case.is_dir():
        msg = f"OpenFOAM case_dir not found: {of_case}"
        raise FileNotFoundError(msg)
    subprocess.run(
        ["bash", "-lc", f"source {of_bashrc} && ./Allclean && ./Allrun"],
        cwd=of_case,
        check=True,
    )


def run_gridfoam(cfg: CompareConfig) -> None:
    if not cfg.gridfoam.enabled:
        return
    gridfoam_script = cfg.case_dir / "gridfoam" / cfg.gridfoam.script
    if not gridfoam_script.is_file():
        msg = f"gridfoam script not found: {gridfoam_script}"
        raise FileNotFoundError(msg)
    subprocess.run(
        ["uv", "run", "python", str(gridfoam_script)],
        check=True,
    )
