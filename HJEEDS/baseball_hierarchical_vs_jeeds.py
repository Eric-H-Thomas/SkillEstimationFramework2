# This file was written or edited by AI and still requires human review. Delete this comment when done.
# Baseball HJEEDS entry point — Statcast real-data hierarchical vs JEEDS comparison.
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS.baseball_config import (
    build_baseball_config_from_args,
    parse_baseball_args,
    print_baseball_dry_run_summary,
)
from HJEEDS.config import planned_output_paths
from HJEEDS.baseball_pipeline import run_single_baseball_seed
from HJEEDS.models import StatcastAgentResult

BASEBALL_AGENT_LEVEL_HEADER = [
    "seed",
    "environment",
    "agent_id",
    "pitcher_id",
    "pitch_type",
    "count_bucket",
    "num_observations",
    "jeeds_posterior_mean_sigma",
    "jeeds_posterior_mean_log_lambda",
    "jeeds_map_sigma",
    "jeeds_map_log_lambda",
    "jeeds_status",
    "hierarchical_posterior_mean_sigma",
    "hierarchical_posterior_mean_log_lambda",
    "hierarchical_map_sigma",
    "hierarchical_map_log_lambda",
    "hierarchical_status",
    "notes",
]


def _value_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _statcast_result_to_row(result: StatcastAgentResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "environment": "baseball",
        "agent_id": result.agent_id,
        "pitcher_id": result.pitcher_id,
        "pitch_type": result.pitch_type,
        "count_bucket": result.count_bucket,
        "num_observations": result.num_observations,
        "jeeds_posterior_mean_sigma": _value_or_blank(result.jeeds.posterior_mean_sigma),
        "jeeds_posterior_mean_log_lambda": _value_or_blank(result.jeeds.posterior_mean_log_lambda),
        "jeeds_map_sigma": _value_or_blank(result.jeeds.map_sigma),
        "jeeds_map_log_lambda": _value_or_blank(result.jeeds.map_log_lambda),
        "jeeds_status": result.jeeds.status,
        "hierarchical_posterior_mean_sigma": _value_or_blank(result.hierarchical.posterior_mean_sigma),
        "hierarchical_posterior_mean_log_lambda": _value_or_blank(
            result.hierarchical.posterior_mean_log_lambda
        ),
        "hierarchical_map_sigma": _value_or_blank(result.hierarchical.map_sigma),
        "hierarchical_map_log_lambda": _value_or_blank(result.hierarchical.map_log_lambda),
        "hierarchical_status": result.hierarchical.status,
        "notes": result.notes,
    }


def write_statcast_agent_level_csv(
    output_path: Path,
    agent_results: Sequence[StatcastAgentResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASEBALL_AGENT_LEVEL_HEADER)
        writer.writeheader()
        for result in agent_results:
            writer.writerow(_statcast_result_to_row(result))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_baseball_args(argv)
    config = build_baseball_config_from_args(args)

    if config.base.dry_run:
        print_baseball_dry_run_summary(config)
        return 0

    all_results: list[StatcastAgentResult] = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        print(
            f"[hier-baseball] Running seed {seed_index}/{config.base.num_seeds}: {seed}",
            flush=True,
        )
        seed_result = run_single_baseball_seed(config, seed)
        all_results.extend(seed_result.agent_results)

    output_paths = planned_output_paths(config.base.output_dir)
    write_statcast_agent_level_csv(output_paths["agent_level_csv"], all_results)
    print(f"[hier-baseball] Wrote results to {config.base.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
