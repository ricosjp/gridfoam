"""Performance configuration, result isolation, and case restoration."""

from pathlib import Path

import pytest
from tests.profile import benchmark_openfoam
from tests.profile.benchmark_results import BenchmarkRow, load_rows, upsert_rows
from tests.profile.case import DEFAULT_CONFIG, available_devices, load_case


def test_case_overrides_do_not_change_source_config() -> None:
    original = DEFAULT_CONFIG.read_bytes()
    cpu = load_case(DEFAULT_CONFIG, "cpu", resolution=(5, 2, 2), steps=2)
    cuda = load_case(DEFAULT_CONFIG, "cuda", resolution=(5, 2, 2), steps=2)
    assert DEFAULT_CONFIG.read_bytes() == original
    cpu_data = cpu.model_dump(mode="json")
    cuda_data = cuda.model_dump(mode="json")
    assert cpu_data["simulator"].pop("device") == "cpu"
    assert cuda_data["simulator"].pop("device") == "cuda"
    assert cpu_data == cuda_data
    assert cpu.simulator.control.endTime == 2 * cpu.simulator.control.deltaT
    assert cpu.fluxel.mesh_path is not None
    assert cpu.fluxel.mesh_path.is_absolute()


def test_cuda_unavailable_keeps_cpu_but_rejects_cuda_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert available_devices(["cpu", "cuda"]) == ["cpu"]
    with pytest.raises(RuntimeError, match="CUDA"):
        available_devices(["cuda"])


def _row(device: str, end_time: float = 10, elapsed: float = 1) -> BenchmarkRow:
    return BenchmarkRow(
        solver="gridfoam",
        device=device,
        resolution=(5, 2, 2),
        n_cells=100,
        elapsed_s=elapsed,
        end_time=end_time,
        delta_t=1,
        config_id="test",
        cpu_threads=2,
    )


def test_csv_keeps_devices_and_durations_separate(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    upsert_rows(path, [_row("cpu"), _row("cuda"), _row("cpu", end_time=20)])
    upsert_rows(path, [_row("cuda", elapsed=0.5)])
    rows = load_rows(path)
    assert len(rows) == 3
    values = {
        (row["device"], float(row["end_time"])): float(row["elapsed_s"])
        for row in rows
    }
    assert values == {("cpu", 10): 1, ("cuda", 10): 0.5, ("cpu", 20): 1}


def test_old_combined_csv_does_not_guess_gridfoam_device(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resolution_scaling.csv"
    path.write_text(
        "solver,resolution,nx,ny,nz,n_cells,elapsed_s,end_time,delta_t,n_steps,time_per_step_s\n"
        "gridfoam,5x2x2,5,2,2,100,1,10,1,10,0.1\n"
    )
    upsert_rows(path, [_row("cuda")])
    assert {row["device"] for row in load_rows(path)} == {"unknown", "cuda"}


def test_openfoam_restores_case_when_run_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = tmp_path / "system"
    system.mkdir()
    block = system / "blockMeshDict"
    control = system / "controlDict"
    block.write_text("hex (0 1 2 3 4 5 6 7) (10 4 4) simpleGrading\n")
    control.write_text("endTime 100;\ndeltaT 1;\n")
    original_block, original_control = block.read_bytes(), control.read_bytes()
    monkeypatch.setattr(benchmark_openfoam, "OF_CASE_DIR", tmp_path)

    def fake_which(_command: str) -> str:
        return "/bin/blockMesh"

    monkeypatch.setattr(benchmark_openfoam.shutil, "which", fake_which)

    def fail_run(*_args: object, **_kwargs: object) -> None:
        assert "(5 2 2)" in block.read_text()
        assert "endTime         2;" in control.read_text()
        raise RuntimeError("test execution failure")

    monkeypatch.setattr(benchmark_openfoam.subprocess, "run", fail_run)
    with pytest.raises(RuntimeError, match="test execution failure"):
        benchmark_openfoam.run_openfoam_benchmark(
            [(5, 2, 2)],
            end_time=None,
            steps=2,
            skip_existing=False,
            csv_path=tmp_path / "results.csv",
        )
    assert block.read_bytes() == original_block
    assert control.read_bytes() == original_control
