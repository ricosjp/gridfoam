from __future__ import annotations

import os
import subprocess
import sys

import pytest

from gridfoam.runtime_config import (
    RUNTIME_TYPE_CHECKS_ENV,
    runtime_type_checks_enabled,
)


def test_runtime_type_checks_enabled_defaults_to_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RUNTIME_TYPE_CHECKS_ENV, raising=False)
    assert runtime_type_checks_enabled()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
def test_runtime_type_checks_enabled_accepts_true_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(RUNTIME_TYPE_CHECKS_ENV, value)
    assert runtime_type_checks_enabled()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled"])
def test_runtime_type_checks_enabled_accepts_false_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(RUNTIME_TYPE_CHECKS_ENV, value)
    assert not runtime_type_checks_enabled()


def test_runtime_type_checks_enabled_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(RUNTIME_TYPE_CHECKS_ENV, "maybe")
    with pytest.raises(ValueError, match=RUNTIME_TYPE_CHECKS_ENV):
        runtime_type_checks_enabled()


def test_runtime_type_checks_env_disables_beartype_import_hook() -> None:
    code = """
from unittest.mock import MagicMock

from gridfoam.solvers.factory import create_solver

cfg = MagicMock()
cfg.method = "not_a_solver"

try:
    create_solver(cfg)
except ValueError:
    pass
else:
    raise SystemExit("expected ValueError")
"""
    env = os.environ.copy()
    env[RUNTIME_TYPE_CHECKS_ENV] = "0"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
