# Paper correspondence: Main `subsec:two_d_darts`.
"""Regression tests for 2D simulation randomness and cluster completeness."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from Environments.Darts.RandomDarts import two_d_darts
from HJEEDS.aggregate_cluster_seeds import aggregate_group, _validate_seed_agent_coverage
from HJEEDS.artifacts import write_agent_level_csv
from HJEEDS.config import AGENT_LEVEL_FILENAME
from HJEEDS.models import AgentResult, MethodEstimate
from HJEEDS.plot_main_paper_higher_dimensional import (
    IntervalSeries,
    _interval_axis_limits,
    load_two_d_series,
)
from HJEEDS.two_d_completion import (
    TWO_D_COMPUTATION_SOURCE_PATHS,
    TWO_D_COMPLETION_FILENAME,
    begin_two_d_part_metadata,
    begin_two_d_aggregation_metadata,
    build_current_two_d_source_provenance,
    finalize_two_d_part_metadata,
    paper_two_d_effective_launch,
    paper_two_d_experiment_argv,
    resolve_two_d_effective_launch,
    validate_complete_two_d_run_metadata,
)
from scripts.runners.run_publication_bench import validate_two_d_output


VALID_POPULATION_FIT_NOTES = (
    "Agent result contains standalone JEEDS and hierarchical estimates. "
    "population_fit: converged=True; selected=optimizer; iterations=4"
)


class TwoDProvenanceInventoryTests(unittest.TestCase):
    def test_transitive_environment_and_completion_contract_are_fingerprinted(self) -> None:
        self.assertIn("HJEEDS/darts_environment.py", TWO_D_COMPUTATION_SOURCE_PATHS)
        self.assertIn("HJEEDS/two_d_completion.py", TWO_D_COMPUTATION_SOURCE_PATHS)

    def test_drifted_effective_worker_argument_cannot_be_sealed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            part_dir = Path(temp_dir) / "part_0"
            argv = paper_two_d_experiment_argv(
                part_dir,
                seed_start=1000,
                num_seeds=1,
            )
            delta_index = argv.index("--delta") + 1
            argv[delta_index] = "10.0"
            drifted_launch = resolve_two_d_effective_launch(argv)
            with self.assertRaisesRegex(ValueError, "effective 2D worker configuration"):
                begin_two_d_part_metadata(
                    part_dir,
                    expected_seed_start=1000,
                    expected_num_seeds=1,
                    expected_agents_per_seed=25,
                    effective_launch_configuration=drifted_launch,
                    require_paper_configuration=True,
                )
            self.assertFalse(part_dir.exists())

    def test_drifted_effective_worker_argument_is_allowed_without_paper_guard(self) -> None:
        with TemporaryDirectory() as temp_dir:
            part_dir = Path(temp_dir) / "part_0"
            argv = paper_two_d_experiment_argv(
                part_dir,
                seed_start=1000,
                num_seeds=1,
            )
            delta_index = argv.index("--delta") + 1
            argv[delta_index] = "10.0"
            drifted_launch = resolve_two_d_effective_launch(argv)
            begin_two_d_part_metadata(
                part_dir,
                expected_seed_start=1000,
                expected_num_seeds=1,
                expected_agents_per_seed=25,
                effective_launch_configuration=drifted_launch,
            )
            self.assertTrue((part_dir / "two_d_part_metadata.json").is_file())


def _ok_estimate(method_name: str) -> MethodEstimate:
    return MethodEstimate(
        method_name=method_name,
        posterior_mean_sigma=12.0,
        posterior_mean_log_lambda=0.0,
        map_sigma=12.0,
        map_log_lambda=0.0,
        rationality_percent=50.0,
        status="ok",
    )


def _agent_result(seed: int, agent_id: int) -> AgentResult:
    return AgentResult(
        seed=seed,
        agent_id=agent_id,
        count_bucket=5,
        num_observations=5,
        sigma_true=12.0,
        log_lambda_true=0.0,
        rationality_percent_true=50.0,
        jeeds=_ok_estimate("jeeds"),
        hierarchical=_ok_estimate("hierarchical"),
        notes=VALID_POPULATION_FIT_NOTES,
    )


def _paper_agent_result(seed: int, agent_id: int) -> AgentResult:
    count_bucket = (5, 10, 25, 100, 1000)[agent_id // 5]
    return replace(
        _agent_result(seed, agent_id),
        count_bucket=count_bucket,
        num_observations=count_bucket,
    )


class TwoDNoiseTests(unittest.TestCase):
    def _draw_sequence(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.asarray(
            [
                two_d_darts.sample_noisy_action(
                    rng,
                    S=(),
                    X=10.0,
                    a=(0.0, 0.0),
                )
                for _ in range(5)
            ]
        )

    def test_successive_throws_advance_rng_but_seed_replays_sequence(self) -> None:
        first_sequence = self._draw_sequence(123)
        replayed_sequence = self._draw_sequence(123)

        np.testing.assert_array_equal(first_sequence, replayed_sequence)
        self.assertGreater(
            np.unique(first_sequence, axis=0).shape[0],
            1,
            "Successive 2D throws must not reuse a frozen execution-noise draw.",
        )


class TwoDGeometryCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        two_d_darts._clear_geometry_caches()

    def tearDown(self) -> None:
        two_d_darts._clear_geometry_caches()

    def test_geometry_is_keyed_by_resolution(self) -> None:
        # Bull value followed by the 20 numbered wedges, with the starting
        # wedge repeated at the end for the legacy angle-index convention.
        slices = (25, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, 6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11)
        fine_coordinates, _ = two_d_darts.get_scores(slices, resolution=5.0)
        coarse_coordinates, _ = two_d_darts.get_scores(slices, resolution=10.0)
        fine_coordinates_again, _ = two_d_darts.get_scores(slices, resolution=5.0)

        self.assertEqual(fine_coordinates.shape, (69 * 69, 2))
        self.assertEqual(coarse_coordinates.shape, (35 * 35, 2))
        self.assertIs(fine_coordinates, fine_coordinates_again)
        self.assertEqual(set(two_d_darts._SCORE_GEOMETRY_CACHE), {5.0, 10.0})

    def test_noise_geometry_is_also_keyed_by_resolution(self) -> None:
        rng = np.random.default_rng(5)
        fine_coordinates, _ = two_d_darts.get_symmetric_normal_distribution(
            rng, XS=12.0, resolution=5.0
        )
        coarse_coordinates, _ = two_d_darts.get_symmetric_normal_distribution(
            rng, XS=12.0, resolution=10.0
        )

        self.assertEqual(fine_coordinates.shape, (137 * 137, 2))
        self.assertEqual(coarse_coordinates.shape, (69 * 69, 2))
        self.assertEqual(set(two_d_darts._NOISE_GEOMETRY_CACHE), {5.0, 10.0})


class TwoDPlotLimitTests(unittest.TestCase):
    def test_limits_are_data_driven_and_include_every_confidence_interval(self) -> None:
        series = {
            "jeeds": IntervalSeries(
                means=(2.0, 3.0),
                lower=(1.5, 2.4),
                upper=(2.7, 3.8),
            ),
            "hierarchical": IntervalSeries(
                means=(1.2, 2.1),
                lower=(0.8, 1.7),
                upper=(1.7, 2.9),
            ),
        }
        lower, upper = _interval_axis_limits(series)
        self.assertLess(lower, 0.8)
        self.assertGreater(upper, 3.8)

        shifted = {
            method: IntervalSeries(
                means=tuple(value + 100.0 for value in values.means),
                lower=tuple(value + 100.0 for value in values.lower),
                upper=tuple(value + 100.0 for value in values.upper),
            )
            for method, values in series.items()
        }
        shifted_lower, shifted_upper = _interval_axis_limits(shifted)
        self.assertGreater(shifted_lower, upper)
        self.assertGreater(shifted_upper, upper)


class ClusterCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.complete_rows = [
            _agent_result(seed, agent_id)
            for seed in (1000, 1001)
            for agent_id in (0, 1)
        ]

    def _validate(self, rows, **kwargs) -> None:
        _validate_seed_agent_coverage(
            rows,
            expected_seed_start=1000,
            expected_num_seeds=2,
            expected_agents_per_seed=2,
            **kwargs,
        )

    def test_complete_seed_agent_grid_is_accepted(self) -> None:
        self._validate(self.complete_rows)

    def test_duplicate_seed_agent_row_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate seed-agent rows"):
            self._validate([*self.complete_rows, self.complete_rows[0]])

    def test_missing_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Seed coverage mismatch"):
            self._validate(self.complete_rows[:2])

    def test_unexpected_seed_is_rejected(self) -> None:
        unexpected = [*self.complete_rows, _agent_result(1002, 0)]
        with self.assertRaisesRegex(ValueError, "Seed coverage mismatch"):
            self._validate(unexpected)

    def test_missing_or_unexpected_agent_is_rejected(self) -> None:
        malformed = [
            row
            for row in self.complete_rows
            if not (row.seed == 1001 and row.agent_id == 1)
        ]
        malformed.append(_agent_result(1001, 2))
        with self.assertRaisesRegex(ValueError, "Agent coverage mismatch"):
            self._validate(malformed)

    def test_estimator_failure_is_rejected(self) -> None:
        failed_row = AgentResult(
            seed=1000,
            agent_id=0,
            count_bucket=5,
            num_observations=5,
            sigma_true=12.0,
            log_lambda_true=0.0,
            rationality_percent_true=50.0,
            jeeds=MethodEstimate(method_name="jeeds", status="failed"),
            hierarchical=_ok_estimate("hierarchical"),
        )
        rows = [failed_row, *self.complete_rows[1:]]
        with self.assertRaisesRegex(ValueError, "Estimator failures"):
            self._validate(rows)

    def test_nonconverged_population_fit_is_rejected(self) -> None:
        nonconverged = AgentResult(
            seed=1000,
            agent_id=0,
            count_bucket=5,
            num_observations=5,
            sigma_true=12.0,
            log_lambda_true=0.0,
            rationality_percent_true=50.0,
            jeeds=_ok_estimate("jeeds"),
            hierarchical=_ok_estimate("hierarchical"),
            notes="population_fit: converged=False; selected=initial",
        )
        rows = [nonconverged, *self.complete_rows[1:]]
        with self.assertRaisesRegex(ValueError, "population-fit diagnostic"):
            self._validate(rows, require_optimizer_selected_population_fit=True)

    def test_nonconverged_population_fit_is_allowed_without_paper_guard(self) -> None:
        nonconverged = AgentResult(
            seed=1000,
            agent_id=0,
            count_bucket=5,
            num_observations=5,
            sigma_true=12.0,
            log_lambda_true=0.0,
            rationality_percent_true=50.0,
            jeeds=_ok_estimate("jeeds"),
            hierarchical=_ok_estimate("hierarchical"),
            notes="population_fit: converged=False; selected=initial",
        )
        rows = [nonconverged, *self.complete_rows[1:]]
        with mock.patch("HJEEDS.aggregate_cluster_seeds.warnings.warn") as warn:
            self._validate(rows)
        warn.assert_called_once()

    def test_success_status_with_missing_metrics_is_rejected(self) -> None:
        incomplete_row = AgentResult(
            seed=1000,
            agent_id=0,
            count_bucket=5,
            num_observations=5,
            sigma_true=12.0,
            log_lambda_true=0.0,
            rationality_percent_true=50.0,
            jeeds=MethodEstimate(method_name="jeeds", status="ok"),
            hierarchical=_ok_estimate("hierarchical"),
            notes=VALID_POPULATION_FIT_NOTES,
        )
        rows = [incomplete_row, *self.complete_rows[1:]]
        with self.assertRaisesRegex(ValueError, "missing publication metrics"):
            self._validate(rows)

    def test_missing_true_decision_metric_is_rejected(self) -> None:
        missing_truth_row = AgentResult(
            seed=1000,
            agent_id=0,
            count_bucket=5,
            num_observations=5,
            sigma_true=12.0,
            log_lambda_true=0.0,
            rationality_percent_true=None,
            jeeds=_ok_estimate("jeeds"),
            hierarchical=_ok_estimate("hierarchical"),
            notes=VALID_POPULATION_FIT_NOTES,
        )
        rows = [missing_truth_row, *self.complete_rows[1:]]
        with self.assertRaisesRegex(ValueError, "Missing true decision-skill percentage"):
            self._validate(rows)


class ClusterAggregationIntegrationTests(unittest.TestCase):
    def _aggregate_tiny_group(self, root: Path, *, cleanup: bool = False) -> Path:
        group_dir = root / "cluster_0"
        for part_index, seed in enumerate((1000, 1001)):
            part_dir = group_dir / f"part_{part_index}"
            begin_two_d_part_metadata(
                part_dir,
                expected_seed_start=seed,
                expected_num_seeds=1,
                expected_agents_per_seed=25,
                effective_launch_configuration=paper_two_d_effective_launch(
                    part_dir,
                    seed_start=seed,
                    num_seeds=1,
                ),
            )
            rows = [_paper_agent_result(seed, agent_id) for agent_id in range(25)]
            write_agent_level_csv(
                part_dir / AGENT_LEVEL_FILENAME,
                rows,
                environment="2d",
            )
            finalize_two_d_part_metadata(part_dir)

        aggregate_group(
            group_dir,
            parts_per_group=2,
            cleanup=cleanup,
            expected_seed_start=1000,
            expected_num_seeds=2,
            expected_agents_per_seed=25,
        )
        return group_dir

    def test_complete_parts_are_aggregated_after_strict_validation(self) -> None:
        with TemporaryDirectory() as temp_dir:
            group_dir = self._aggregate_tiny_group(Path(temp_dir))

            combined_csv = group_dir / AGENT_LEVEL_FILENAME
            self.assertTrue(combined_csv.is_file())
            self.assertEqual(len(combined_csv.read_text().splitlines()), 51)
            completion = validate_complete_two_d_run_metadata(group_dir)
            self.assertEqual(completion["completion_status"], "complete")
            self.assertEqual(completion["expected_seeds"], [1000, 1001])
            self.assertEqual(completion["expected_agent_ids"], list(range(25)))
            self.assertEqual(
                set(completion["artifact_sha256"]),
                {
                    "agent_level_results.csv",
                    "summary_by_bucket.csv",
                    "summary_overall.csv",
                    "error_by_count_bucket.png",
                },
            )

    def test_failed_aggregation_leaves_explicit_incomplete_record(self) -> None:
        with TemporaryDirectory() as temp_dir:
            group_dir = Path(temp_dir) / "cluster_0"
            with self.assertRaisesRegex(FileNotFoundError, "Missing part directory"):
                aggregate_group(
                    group_dir,
                    parts_per_group=1,
                    cleanup=False,
                    expected_seed_start=1000,
                    expected_num_seeds=1,
                    expected_agents_per_seed=2,
                )
            self.assertTrue((group_dir / TWO_D_COMPLETION_FILENAME).is_file())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_complete_two_d_run_metadata(group_dir)

    def test_pre_fix_part_without_provenance_cannot_be_reaggregated(self) -> None:
        with TemporaryDirectory() as temp_dir:
            group_dir = Path(temp_dir) / "cluster_0"
            part_dir = group_dir / "part_0"
            write_agent_level_csv(
                part_dir / AGENT_LEVEL_FILENAME,
                [_agent_result(1000, agent_id) for agent_id in (0, 1)],
                environment="2d",
            )
            with self.assertRaisesRegex(FileNotFoundError, "part provenance"):
                aggregate_group(
                    group_dir,
                    parts_per_group=1,
                    cleanup=False,
                    expected_seed_start=1000,
                    expected_num_seeds=1,
                    expected_agents_per_seed=2,
                )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_complete_two_d_run_metadata(group_dir)

    def test_modified_completed_artifact_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            group_dir = self._aggregate_tiny_group(Path(temp_dir))
            summary = group_dir / "summary_by_bucket.csv"
            with summary.open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "artifact size changed"):
                validate_complete_two_d_run_metadata(group_dir)

    def test_stale_computational_source_fingerprint_is_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            group_dir = self._aggregate_tiny_group(Path(temp_dir))
            stale = build_current_two_d_source_provenance()
            stale = {**stale, "fingerprint_sha256": "0" * 64}
            with mock.patch(
                "HJEEDS.two_d_completion.build_current_two_d_source_provenance",
                return_value=stale,
            ):
                with self.assertRaisesRegex(ValueError, "source provenance is stale"):
                    validate_complete_two_d_run_metadata(
                        group_dir,
                        require_paper_configuration=True,
                    )

    def test_stale_computational_source_fingerprint_is_allowed_without_paper_guard(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            group_dir = self._aggregate_tiny_group(Path(temp_dir))
            stale = build_current_two_d_source_provenance()
            stale = {**stale, "fingerprint_sha256": "0" * 64}
            with mock.patch(
                "HJEEDS.two_d_completion.build_current_two_d_source_provenance",
                return_value=stale,
            ):
                completion = validate_complete_two_d_run_metadata(group_dir)
            self.assertEqual(completion["completion_status"], "complete")

    def test_noncanonical_seed_agent_bundle_is_rejected_by_paper_guard(self) -> None:
        with TemporaryDirectory() as temp_dir:
            group_dir = self._aggregate_tiny_group(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "not the canonical paper run"):
                validate_complete_two_d_run_metadata(
                    group_dir,
                    require_paper_configuration=True,
                )

    def test_optional_cleanup_failure_is_nonfatal_after_sealing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with mock.patch(
                "HJEEDS.aggregate_cluster_seeds.shutil.rmtree",
                side_effect=OSError("simulated cleanup failure"),
            ):
                group_dir = self._aggregate_tiny_group(Path(temp_dir), cleanup=True)
            completion = validate_complete_two_d_run_metadata(group_dir)
            self.assertEqual(completion["completion_status"], "complete")


class TwoDPaperPlotCompletionGuardTests(unittest.TestCase):
    def test_plot_loader_rejects_summary_without_completion_record(self) -> None:
        with TemporaryDirectory() as temp_dir:
            summary = Path(temp_dir) / "summary_by_bucket.csv"
            summary.write_text("method,metric,count_bucket,num_agents,mean,ci_lower,ci_upper,notes\n")
            with self.assertRaisesRegex(FileNotFoundError, "completion metadata"):
                load_two_d_series(summary, require_paper_configuration=True)

    def test_explicit_incomplete_record_is_rejected_before_plot_csv_is_read(self) -> None:
        with TemporaryDirectory() as temp_dir:
            group_dir = Path(temp_dir)
            begin_two_d_aggregation_metadata(
                group_dir,
                expected_seed_start=1000,
                expected_num_seeds=500,
                expected_agents_per_seed=25,
                parts_per_group=10,
            )
            summary = group_dir / "summary_by_bucket.csv"
            summary.write_text("not,a,valid,summary\n")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_two_d_series(summary, require_paper_configuration=True)


class LocalPublicationBenchValidationTests(unittest.TestCase):
    def test_local_two_d_output_requires_complete_converged_population_fits(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            rows = [_agent_result(1000, agent_id) for agent_id in range(25)]
            output_path = output_root / "two_d" / AGENT_LEVEL_FILENAME
            write_agent_level_csv(output_path, rows, environment="2d")

            validate_two_d_output(output_root, first_seed=1000, num_seeds=1)

            rows[0] = replace(
                rows[0],
                notes="population_fit: converged=False; selected=initial",
            )
            write_agent_level_csv(output_path, rows, environment="2d")
            with self.assertRaisesRegex(ValueError, "population-fit diagnostic"):
                validate_two_d_output(output_root, first_seed=1000, num_seeds=1)


if __name__ == "__main__":
    unittest.main()
