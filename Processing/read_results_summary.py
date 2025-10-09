from __future__ import annotations

"""Summarise darts experiment `.results` files on the command line.

Example usage
-------------
python Processing/read_results_summary.py Experiments/1d/testResultsFolder/results
python Processing/read_results_summary.py Experiments/1d/testResultsFolder/results/<file>.results
"""

import argparse
import pickle
from pathlib import Path
from statistics import StatisticsError, mean
from typing import Iterable, Mapping, MutableMapping


def _numeric_stats(values: Iterable[float]) -> str:
    """Return a compact "count/mean/min/max" summary for numeric iterables."""
    data = []
    for value in values:
        if isinstance(value, (int, float)):
            data.append(float(value))
    if not data:
        return "(no numeric data)"
    try:
        avg = mean(data)
    except StatisticsError:
        avg = float("nan")
    return f"count={len(data)} mean={avg:.3f} min={min(data):.3f} max={max(data):.3f}"


def _flatten_nested_lists(nested):
    for element in nested:
        if isinstance(element, (list, tuple)):
            yield from _flatten_nested_lists(element)
        else:
            yield element


def describe_results(data: Mapping[str, object]) -> MutableMapping[str, object]:
    """Build a human-friendly summary dictionary for a darts results payload."""
    summary: MutableMapping[str, object] = {}
    summary["agent_name"] = data.get("agent_name")
    summary["xskill"] = data.get("xskill")
    summary["numObservations"] = data.get("numObservations")
    summary["delta"] = data.get("delta")
    summary["estimators"] = data.get("estimators_list")
    summary["expTotalTime"] = data.get("expTotalTime")
    summary["lastEdited"] = data.get("lastEdited")

    observed = data.get("observed_rewards", [])
    summary["observed_rewards"] = _numeric_stats(observed)

    intended = data.get("intended_actions", [])
    summary["intended_actions"] = (
        f"count={len(intended)} sample={intended[:3]}"
        if isinstance(intended, list)
        else "(not a list)"
    )

    noisy = data.get("noisy_actions", [])
    summary["noisy_actions"] = (
        f"count={len(noisy)} sample={noisy[:3]}"
        if isinstance(noisy, list)
        else "(not a list)"
    )

    exp_rewards = data.get("exp_rewards", [])
    summary["exp_rewards"] = _numeric_stats(exp_rewards)

    rs_rewards = data.get("rs_rewards", [])
    if isinstance(rs_rewards, list):
        summary["rs_rewards"] = _numeric_stats(_flatten_nested_lists(rs_rewards))
    else:
        summary["rs_rewards"] = "(not a list)"

    return summary


def load_pickle(path: Path) -> Mapping[str, object]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def summarize_path(path: Path) -> None:
    if path.is_dir():
        files = sorted(p for p in path.glob("*.results"))
        if not files:
            print(f"No .results files found in {path}")
            return
        for file_path in files:
            print(f"\n=== {file_path.name} ===")
            summarize_file(file_path)
    else:
        summarize_file(path)


def summarize_file(path: Path) -> None:
    try:
        data = load_pickle(path)
    except Exception as exc:  # pragma: no cover - best effort diagnostic helper
        print(f"Failed to read {path}: {exc}")
        return

    summary = describe_results(data)
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Show all other keys so the user can explore them manually
    remaining = sorted(set(data.keys()) - set(summary.keys()))
    if remaining:
        print("other keys:")
        for key in remaining:
            print(f"  - {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .results file or a directory that contains results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize_path(args.path)


if __name__ == "__main__":
    main()
