"""Inference-mode wrapping for FV evaluations that need tensor versions."""

from collections.abc import Callable

import torch


def call_without_inference[T](fn: Callable[[], T]) -> T:
    """Run ``fn`` with ordinary tensors when inference mode is active.

    FV cache tokens require version counters, so inference mode is disabled
    and replaced by ``no_grad`` for the duration of ``fn``.
    """
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False), torch.no_grad():
            return fn()
    return fn()
