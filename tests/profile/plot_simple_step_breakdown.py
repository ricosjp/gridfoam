"""
Visualize SIMPLE.step execution-time breakdown from pyinstrument HTML profiles.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROFILE_DIR = Path(__file__).resolve().parent
DEFAULT_CPU_HTML = PROFILE_DIR / "outputs" / "time" / "cpu" / "profile.html"
DEFAULT_CUDA_HTML = PROFILE_DIR / "outputs" / "time" / "cuda" / "profile.html"
DEFAULT_OUTPUT = PROFILE_DIR / "outputs" / "time" / "simple_step_breakdown.png"


def _load_pyinstrument_session(path: Path) -> dict[str, Any]:
    text = path.read_text()
    match = re.search(r"const sessionData = (\{.*\});", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find sessionData in {path}")
    return json.loads(match.group(1))


def _short_label(identifier: str) -> str:
    parts = identifier.split("\x00")
    name = parts[0] or "[unknown]"
    path = parts[1] if len(parts) > 1 else ""
    module = Path(path).name if path else ""
    if name == "[self]":
        return "step [self]"
    if module:
        return f"{name} ({module})"
    return name


def _find_simple_steps(frame_tree: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        identifier = node.get("identifier", "")
        if identifier.startswith("step") and "simple.py" in identifier:
            steps.append(node)
        for child in node.get("children", []):
            walk(child)

    walk(frame_tree)
    if not steps:
        raise ValueError("No SIMPLE.step frames found in profile")
    return steps


def _aggregate_step_children(
    steps: list[dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    step_total = sum(step.get("time", 0.0) for step in steps)

    for step in steps:
        for child in step.get("children", []):
            totals[_short_label(child["identifier"])] += child.get("time", 0.0)

    return step_total, dict(totals)


def _collapse_small_entries(
    entries: dict[str, float],
    *,
    min_fraction: float,
) -> dict[str, float]:
    if not entries:
        return {}

    total = sum(entries.values())
    if total <= 0.0:
        return entries

    kept: dict[str, float] = {}
    other = 0.0
    threshold = total * min_fraction
    for label, value in entries.items():
        if value < threshold:
            other += value
        else:
            kept[label] = value

    if other > 0.0:
        kept["other"] = other
    return kept


def extract_simple_step_breakdown(path: Path) -> tuple[float, dict[str, float]]:
    session = _load_pyinstrument_session(path)
    steps = _find_simple_steps(session["frame_tree"])
    return _aggregate_step_children(steps)


def _plot_breakdown(
    ax: plt.Axes,
    *,
    title: str,
    step_total: float,
    entries: dict[str, float],
) -> None:
    labels = list(entries.keys())
    values = [entries[label] for label in labels]
    percentages = [100.0 * value / step_total for value in values]

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color="C0", alpha=0.85)
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Elapsed time [s]")
    ax.set_title(f"{title}\nSIMPLE.step total: {step_total:.2f} s")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    for bar, pct in zip(bars, percentages, strict=True):
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {pct:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
        )


def plot_simple_step_breakdown(
    cpu_html: Path,
    cuda_html: Path,
    output_path: Path,
    *,
    min_fraction: float,
) -> None:
    profiles = {
        "CPU profile": cpu_html,
        "CUDA profile": cuda_html,
    }

    fig, axes = plt.subplots(
        1,
        len(profiles),
        figsize=(14, 7),
        constrained_layout=True,
    )
    if len(profiles) == 1:
        axes = [axes]

    for ax, (title, path) in zip(axes, profiles.items(), strict=True):
        step_total, entries = extract_simple_step_breakdown(path)
        collapsed = _collapse_small_entries(entries, min_fraction=min_fraction)
        ordered = dict(
            sorted(collapsed.items(), key=lambda item: item[1], reverse=True)
        )
        _plot_breakdown(ax, title=title, step_total=step_total, entries=ordered)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-html", type=Path, default=DEFAULT_CPU_HTML)
    parser.add_argument("--cuda-html", type=Path, default=DEFAULT_CUDA_HTML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.01,
        help=(
            "Group entries below this fraction of SIMPLE.step time into 'other'"
        ),
    )
    args = parser.parse_args()
    plot_simple_step_breakdown(
        args.cpu_html,
        args.cuda_html,
        args.output,
        min_fraction=args.min_fraction,
    )


if __name__ == "__main__":
    main()
