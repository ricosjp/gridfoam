from __future__ import annotations

import logging
import logging.config
from pathlib import Path

_DEFAULT_CONF_PATH = Path(__file__).with_name("logging.conf")


def configure_logging(
    *,
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    conf_path: str | Path | None = None,
) -> None:
    """
    Configure gridfoam logging for runtime diagnostics.

    Parameters
    ----------
    level : int, optional
        Base logging level.
    log_file : str | Path | None, optional
        Optional file path for persistent logs.
    conf_path : str | Path | None, optional
        Path to logging configuration file. If omitted, uses
        ``src/gridfoam/logging.conf``.
    """
    if conf_path is not None:
        config_path = Path(conf_path)
    else:
        config_path = _DEFAULT_CONF_PATH
    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False,
    )

    root = logging.getLogger("gridfoam")
    root.setLevel(level)

    if log_file is not None:
        formatter = logging.Formatter(
            fmt=(
                "[%(asctime)s] [%(levelname)s] [%(process)d] "
                "[%(name)s] [%(funcName)s] [%(lineno)d] %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(Path(log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
