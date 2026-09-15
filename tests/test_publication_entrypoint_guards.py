# Paper correspondence: Main `sec:experiments`; publication entry points and plot guards.
"""Regression tests for publication-runner and main-baseline plot guards."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from HJEEDS import plot_main_paper_baseline as baseline_plot
from scripts.runners import run_publication_bench as publication_bench


class PublicationBenchBaseballGuardTests(unittest.TestCase):
    def test_default_local_components_are_synthetic_only(self) -> None:
        args = publication_bench.parse_args([])
        self.assertEqual(args.components, ("1d", "2d"))

    def test_baseball_component_never_builds_a_local_compute_command(self) -> None:
        args = publication_bench.parse_args(["--components", "baseball", "--dry-run"])
        commands = publication_bench.build_commands(args, Path("/tmp/unused-output"))
        self.assertEqual(commands["baseball"], [])

    def test_baseball_audit_calls_both_completion_and_cohort_validators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            completion = {"completion_status": "complete"}
            with mock.patch.object(
                publication_bench,
                "validate_complete_convergence_run_metadata",
                return_value=completion,
            ) as validate_completion, mock.patch.object(
                publication_bench,
                "validate_paper_bbip20_cohort",
            ) as validate_cohort:
                actual = publication_bench.validate_baseball_paper_endpoint(results_dir)

            self.assertIs(actual, completion)
            validate_completion.assert_called_once_with(results_dir.resolve())
            validate_cohort.assert_called_once_with(results_dir.resolve(), completion)

    def test_missing_baseball_endpoint_is_not_reported_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "bench"
            with mock.patch.object(
                publication_bench,
                "validate_baseball_paper_endpoint",
                side_effect=FileNotFoundError("missing completion record"),
            ):
                return_code = publication_bench.main(
                    [
                        "--components",
                        "baseball",
                        "--output-root",
                        str(output_root),
                        "--baseball-results-dir",
                        str(Path(temporary_directory) / "missing-results"),
                    ]
                )

            self.assertEqual(return_code, 1)
            manifest = json.loads(
                (output_root / "publication_bench_manifest.json").read_text(encoding="utf-8")
            )
            status = manifest["component_status"]["baseball"]
            self.assertIn("not supported by the local bench", status)
            self.assertNotEqual(status, "complete")

    def test_valid_baseball_endpoint_is_labeled_as_external_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "bench"
            with mock.patch.object(
                publication_bench,
                "validate_baseball_paper_endpoint",
                return_value={"completion_status": "complete"},
            ):
                return_code = publication_bench.main(
                    [
                        "--components",
                        "baseball",
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(return_code, 0)
            manifest = json.loads(
                (output_root / "publication_bench_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["component_status"]["baseball"],
                "validated external Slurm paper endpoint",
            )
            self.assertNotEqual(manifest["component_status"]["baseball"], "complete")


class BaselinePlotInputGuardTests(unittest.TestCase):
    SUMMARY_COLUMNS = (
        "method",
        "metric",
        "count_bucket",
        "num_agents",
        "mean",
        "ci_lower",
        "ci_upper",
        "notes",
    )
    AGENT_COLUMNS = (
        "seed",
        "environment",
        "agent_id",
        "count_bucket",
        "num_observations",
        "sigma_true",
        "log_lambda_true",
        "rationality_percent_true",
        "jeeds_posterior_mean_sigma",
        "jeeds_posterior_mean_log_lambda",
        "jeeds_rationality_percent",
        "jeeds_status",
        "hierarchical_posterior_mean_sigma",
        "hierarchical_posterior_mean_log_lambda",
        "hierarchical_rationality_percent",
        "hierarchical_status",
        "notes",
    )

    def _summary_rows(self) -> list[dict[str, object]]:
        metric_values = {
            baseline_plot.EXECUTION_METRIC: {"jeeds": 1.0, "hierarchical": 0.5},
            baseline_plot.DECISION_METRIC: {"jeeds": 2.0, "hierarchical": 1.0},
            baseline_plot.RAW_DECISION_METRIC: {"jeeds": 0.4, "hierarchical": 0.2},
        }
        return [
            {
                "method": method,
                "metric": metric,
                "count_bucket": bucket,
                "num_agents": 2,
                "mean": metric_values[metric][method],
                "ci_lower": metric_values[metric][method],
                "ci_upper": metric_values[metric][method],
                "notes": "test",
            }
            for method in sorted(baseline_plot.CANONICAL_METHODS)
            for metric in sorted(baseline_plot.CANONICAL_SUMMARY_METRICS)
            for bucket in baseline_plot.CANONICAL_BUCKETS
        ]

    def _agent_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for seed in (10, 11):
            for agent_id, bucket in enumerate(baseline_plot.CANONICAL_BUCKETS):
                rows.append(
                    {
                        "seed": seed,
                        "environment": "1d",
                        "agent_id": agent_id,
                        "count_bucket": bucket,
                        "num_observations": bucket,
                        "sigma_true": 1.0,
                        "log_lambda_true": 0.0,
                        "rationality_percent_true": 25.0,
                        "jeeds_posterior_mean_sigma": 2.0,
                        "jeeds_posterior_mean_log_lambda": 0.4,
                        "jeeds_rationality_percent": 27.0,
                        "jeeds_status": "ok",
                        "hierarchical_posterior_mean_sigma": 1.5,
                        "hierarchical_posterior_mean_log_lambda": 0.2,
                        "hierarchical_rationality_percent": 26.0,
                        "hierarchical_status": "ok",
                        "notes": (
                            "population_fit: converged=True; selected=optimizer;"
                        ),
                    }
                )
        return rows

    def _write_csv(
        self,
        path: Path,
        columns: tuple[str, ...],
        rows: list[dict[str, object]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_summary_matches_agent_level_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / "summary_by_bucket.csv"
            agent_path = root / "agent_level_results.csv"
            self._write_csv(summary_path, self.SUMMARY_COLUMNS, self._summary_rows())
            self._write_csv(agent_path, self.AGENT_COLUMNS, self._agent_rows())

            rows = baseline_plot.read_summary_rows(summary_path)
            self.assertEqual(len(rows), 20)
            baseline_plot.validate_summary_against_agent_results(
                rows,
                agent_path,
                expected_first_seed=10,
                expected_num_seeds=2,
                expected_agents_per_bucket=1,
            )

    def test_duplicate_summary_cell_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "summary.csv"
            rows = self._summary_rows()
            rows.append(dict(rows[0]))
            self._write_csv(path, self.SUMMARY_COLUMNS, rows)
            with self.assertRaisesRegex(ValueError, "Duplicate baseline summary row"):
                baseline_plot.read_summary_rows(path)

    def test_partial_bucket_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "summary.csv"
            rows = [row for row in self._summary_rows() if row["count_bucket"] != 1000]
            self._write_csv(path, self.SUMMARY_COLUMNS, rows)
            with self.assertRaisesRegex(ValueError, "observation buckets"):
                baseline_plot.read_summary_rows(path)

    def test_nonfinite_summary_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "summary.csv"
            rows = self._summary_rows()
            rows[0]["mean"] = "nan"
            self._write_csv(path, self.SUMMARY_COLUMNS, rows)
            with self.assertRaisesRegex(ValueError, "Non-finite mean"):
                baseline_plot.read_summary_rows(path)

    def test_stale_summary_is_rejected_against_agent_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / "summary_by_bucket.csv"
            agent_path = root / "agent_level_results.csv"
            rows = self._summary_rows()
            for row in rows:
                if (
                    row["method"] == "jeeds"
                    and row["metric"] == baseline_plot.EXECUTION_METRIC
                    and row["count_bucket"] == 5
                ):
                    row["mean"] = 1.1
                    row["ci_upper"] = 1.1
            self._write_csv(summary_path, self.SUMMARY_COLUMNS, rows)
            self._write_csv(agent_path, self.AGENT_COLUMNS, self._agent_rows())

            parsed = baseline_plot.read_summary_rows(summary_path)
            with self.assertRaisesRegex(ValueError, "Stale baseline summary"):
                baseline_plot.validate_summary_against_agent_results(
                    parsed,
                    agent_path,
                    expected_first_seed=10,
                    expected_num_seeds=2,
                    expected_agents_per_bucket=1,
                )

    def test_partial_seed_run_is_rejected_for_paper_plot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / "summary_by_bucket.csv"
            agent_path = root / "agent_level_results.csv"
            self._write_csv(summary_path, self.SUMMARY_COLUMNS, self._summary_rows())
            self._write_csv(agent_path, self.AGENT_COLUMNS, self._agent_rows())

            parsed = baseline_plot.read_summary_rows(summary_path)
            with self.assertRaisesRegex(ValueError, "seed coverage is partial"):
                baseline_plot.validate_summary_against_agent_results(parsed, agent_path)

    def test_structurally_consistent_pre_fix_agent_rows_are_rejected(self) -> None:
        mutations = (
            (
                "missing optimizer diagnostic",
                lambda rows: rows[0].update({"notes": "legacy result"}),
                "population-fit diagnostic",
            ),
            (
                "observation-count mismatch",
                lambda rows: rows[0].update({"num_observations": 999}),
                "differs from num_observations",
            ),
            (
                "out-of-range rationality",
                lambda rows: rows[0].update({"jeeds_rationality_percent": 101.0}),
                "outside \\[0, 100\\]",
            ),
        )
        for label, mutate, expected_message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                summary_path = root / "summary_by_bucket.csv"
                agent_path = root / "agent_level_results.csv"
                agent_rows = self._agent_rows()
                mutate(agent_rows)
                self._write_csv(summary_path, self.SUMMARY_COLUMNS, self._summary_rows())
                self._write_csv(agent_path, self.AGENT_COLUMNS, agent_rows)

                parsed = baseline_plot.read_summary_rows(summary_path)
                with self.assertRaisesRegex(ValueError, expected_message):
                    baseline_plot.validate_summary_against_agent_results(
                        parsed,
                        agent_path,
                        expected_first_seed=10,
                        expected_num_seeds=2,
                        expected_agents_per_bucket=1,
                    )


if __name__ == "__main__":
    unittest.main()
