# Paper correspondence: Main `sec:experiments`; 1D result completeness.
"""Regression tests for fail-fast publication-result validation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.validate_publication_results import (
    CANONICAL_SCENARIO_AGENT_COUNTS,
    REQUIRED_COLUMNS,
    validate_publication_root,
)


def _valid_row(seed: int, agent_id: int) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "seed": seed,
            "agent_id": agent_id,
            "count_bucket": 5,
            "num_observations": 5,
            "sigma_true": 1.5,
            "log_lambda_true": 0.0,
            "rationality_percent_true": 25.0,
            "jeeds_posterior_mean_sigma": 1.6,
            "jeeds_posterior_mean_log_lambda": 0.1,
            "jeeds_rationality_percent": 27.0,
            "jeeds_status": "ok",
            "hierarchical_posterior_mean_sigma": 1.55,
            "hierarchical_posterior_mean_log_lambda": 0.05,
            "hierarchical_rationality_percent": 26.0,
            "hierarchical_status": "ok",
            "notes": (
                "Agent result contains standalone JEEDS and hierarchical empirical-Bayes "
                "estimates. population_fit: converged=True; selected=optimizer;"
            ),
        }
    )
    return row


class PublicationResultValidationTests(unittest.TestCase):
    def _write_scenario(
        self,
        root: Path,
        rows: list[dict[str, object]],
        scenario_dir: str = "scenario",
    ) -> Path:
        path = root / scenario_dir / "agent_level_results.csv"
        path.parent.mkdir(parents=True)
        columns = sorted(REQUIRED_COLUMNS)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_canonical_tree(
        self,
        root: Path,
        *,
        agent_count_overrides: dict[str, int] | None = None,
    ) -> None:
        overrides = agent_count_overrides or {}
        for relative_path, default_num_agents in CANONICAL_SCENARIO_AGENT_COUNTS.items():
            num_agents = overrides.get(relative_path, default_num_agents)
            rows = [_valid_row(10, agent_id) for agent_id in range(num_agents)]
            if "decision_model_rational" in relative_path:
                for row in rows:
                    row["rationality_percent_true"] = 100.0
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["environment", *sorted(REQUIRED_COLUMNS)])
                writer.writeheader()
                writer.writerows({"environment": "1d", **row} for row in rows)

    def test_complete_small_tree_passes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_scenario(root, [_valid_row(10, 0), _valid_row(11, 0)])
            report = validate_publication_root(
                root,
                first_seed=10,
                num_seeds=2,
                expected_scenarios=1,
                expected_rows_per_seed=1,
            )
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["total_rows"], 2)

    def test_failed_estimator_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            row = _valid_row(10, 0)
            row["hierarchical_status"] = "invalid_posterior"
            self._write_scenario(root, [row])
            with self.assertRaisesRegex(ValueError, "hierarchical_status"):
                validate_publication_root(
                    root,
                    first_seed=10,
                    num_seeds=1,
                    expected_scenarios=1,
                    expected_rows_per_seed=1,
                )

    def test_missing_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_scenario(root, [_valid_row(10, 0)])
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                validate_publication_root(
                    root,
                    first_seed=10,
                    num_seeds=2,
                    expected_scenarios=1,
                    expected_rows_per_seed=1,
                )

    def test_nonconverged_population_fit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            row = _valid_row(10, 0)
            row["notes"] = "population_fit: converged=False; selected=initial"
            self._write_scenario(root, [row])
            with self.assertRaisesRegex(ValueError, "population-fit diagnostic"):
                validate_publication_root(
                    root,
                    first_seed=10,
                    num_seeds=1,
                    expected_scenarios=1,
                    expected_rows_per_seed=1,
                )

    def test_converged_fit_that_did_not_select_optimizer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            row = _valid_row(10, 0)
            row["notes"] = "population_fit: converged=True; selected=initial"
            self._write_scenario(root, [row])
            with self.assertRaisesRegex(ValueError, "optimizer-selected"):
                validate_publication_root(
                    root,
                    first_seed=10,
                    num_seeds=1,
                    expected_scenarios=1,
                    expected_rows_per_seed=1,
                )

    def test_out_of_range_rationality_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            row = _valid_row(10, 0)
            row["hierarchical_rationality_percent"] = 100.1
            self._write_scenario(root, [row])
            with self.assertRaisesRegex(ValueError, "outside \\[0, 100\\]"):
                validate_publication_root(
                    root,
                    first_seed=10,
                    num_seeds=1,
                    expected_scenarios=1,
                    expected_rows_per_seed=1,
                )

    def test_rational_policy_requires_one_hundred_percent_truth(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            row = _valid_row(10, 0)
            row["rationality_percent_true"] = 99.0
            self._write_scenario(root, [row], scenario_dir="decision_model_rational")
            with self.assertRaisesRegex(ValueError, "behavioral truth must be 100%"):
                validate_publication_root(
                    root,
                    first_seed=10,
                    num_seeds=1,
                    expected_scenarios=1,
                    expected_rows_per_seed=1,
                )

    def test_canonical_design_rejects_replaced_scenario_with_same_global_totals(self):
        self.assertEqual(len(CANONICAL_SCENARIO_AGENT_COUNTS), 93)
        self.assertEqual(sum(CANONICAL_SCENARIO_AGENT_COUNTS.values()), 2565)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_canonical_tree(root)

            baseline = root / "baseline" / "agent_level_results.csv"
            replacement = root / "wrong_replacement" / "agent_level_results.csv"
            replacement.parent.mkdir(parents=True)
            baseline.replace(replacement)

            with self.assertRaisesRegex(ValueError, "scenario path mismatch"):
                validate_publication_root(root, first_seed=10, num_seeds=1)

    def test_canonical_design_rejects_compensating_per_scenario_agent_counts(self):
        baseline_path = "baseline/agent_level_results.csv"
        outlier_path = "outlier_sensitivity/outliers_000/agent_level_results.csv"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_canonical_tree(
                root,
                agent_count_overrides={
                    baseline_path: 24,
                    outlier_path: 26,
                },
            )

            with self.assertRaisesRegex(ValueError, "per-scenario agent-count mismatch"):
                validate_publication_root(root, first_seed=10, num_seeds=1)


if __name__ == "__main__":
    unittest.main()
