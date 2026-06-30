import logging
import sys
from pathlib import Path

import torch

from gridfoam.io.vtu import save_export_fields_as_vtu
from gridfoam.runner import manual_step

logger = logging.getLogger(__name__)


def configure_run_logger(log_file: Path) -> None:
    formatter = logging.Formatter("%(message)s")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main() -> None:
    log_file = Path(__file__).resolve().parent / "outputs" / "motorBike.log"
    configure_run_logger(log_file)
    config_path = Path(__file__).resolve().parent / "data" / "config.yaml"
    for step_data in manual_step(config_path):
        step_data.algorithm.step()
        write_interval = step_data.control.writeInterval
        end_time = step_data.control.endTime
        deltaT = step_data.control.deltaT
        n_steps = int(end_time / deltaT)
        if step_data.step % write_interval == 0 or step_data.step == n_steps:
            output_dir = Path(step_data.control.output.output_dir)
            base_name = step_data.control.output.base_name
            grid = step_data.algorithm.grid
            save_export_fields_as_vtu(
                grid,
                str(output_dir / f"{base_name}_{step_data.step:04d}.vtu"),
                ugrid=step_data.ugrid,
            )
            U = grid.get_cellfield("U")
            assert U is not None
            max_u = torch.linalg.vector_norm(U.data, ord=2, dim=1).max().item()
            logger.info("step=%4d max|U|=%.4e", step_data.step, max_u)


if __name__ == "__main__":
    main()
