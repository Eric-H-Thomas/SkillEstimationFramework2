# Paper correspondence: Main `subsec:sensitivity_analyses`; supplementary sensitivity sections.
"""Regression tests for the publication bench's one-factor-at-a-time design."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from HJEEDS import darts_agents_per_bucket_sensitivity as agents_per_bucket
from HJEEDS import darts_anchor_availability_sensitivity as anchor_availability
from HJEEDS import darts_compound_stress_sensitivity as compound_stress
from HJEEDS import darts_decision_model_sensitivity as decision_model
from HJEEDS import darts_population_shape_sensitivity as population_shape
from HJEEDS import darts_true_correlation_sensitivity as true_correlation
from HJEEDS import plot_remaining_sensitivity_robustness as remaining_plots
from scripts.runners import run_publication_bench as publication_bench

import run_hjeeds_paper_experiments as paper_runner


def _args(module):
    return module.parse_args(
        [
            "--seed",
            "default",
            "--num-seeds",
            "1",
            "--output-dir",
            "/tmp/hjeeds_experiment_design_test",
        ]
    )


class ExperimentDesignIsolationTests(unittest.TestCase):
    def test_agents_per_bucket_varies_only_population_size(self):
        args = _args(agents_per_bucket)
        configs = tuple(
            agents_per_bucket.build_config_for_agents_per_bucket(args, value)
            for value in agents_per_bucket.DEFAULT_AGENTS_PER_BUCKET_VALUES
        )
        conditions = agents_per_bucket.default_hyperprior_conditions()
        scenarios = agents_per_bucket.build_scenarios(configs, conditions)

        self.assertEqual(len(scenarios), 5)
        self.assertEqual({scenario.condition_slug for scenario in scenarios}, {"default"})
        self.assertEqual(
            {scenario.config.agents_per_bucket for scenario in scenarios},
            set(agents_per_bucket.DEFAULT_AGENTS_PER_BUCKET_VALUES),
        )

    def test_single_factor_studies_use_default_population_size(self):
        shape_scenarios = population_shape.build_scenarios(
            _args(population_shape),
            population_shape.DEFAULT_AGENTS_PER_BUCKET_VALUES,
        )
        decision_scenarios = decision_model.build_scenarios(_args(decision_model))
        correlation_scenarios = true_correlation.build_scenarios(_args(true_correlation))

        expected_counts = (
            (shape_scenarios, 3),
            (decision_scenarios, 4),
            (correlation_scenarios, 5),
        )
        for scenarios, expected_count in expected_counts:
            with self.subTest(expected_count=expected_count):
                self.assertEqual(len(scenarios), expected_count)
                self.assertEqual(
                    {scenario.config.agents_per_bucket for scenario in scenarios},
                    {5},
                )

    def test_compound_stress_combines_factors_at_default_population_size(self):
        scenarios = compound_stress.build_scenarios(_args(compound_stress))

        self.assertEqual(len(scenarios), 3)
        self.assertEqual({scenario.config.agents_per_bucket for scenario in scenarios}, {5})
        combined_settings = {
            (
                scenario.compound_stress.hyperprior_condition_slug,
                scenario.compound_stress.population_shape_slug,
                scenario.compound_stress.decision_model_slug,
                scenario.compound_stress.true_correlation,
            )
            for scenario in scenarios
        }
        self.assertEqual(len(combined_settings), 3)
        by_slug = {scenario.compound_stress.slug: scenario for scenario in scenarios}
        moderate = by_slug["moderate_compound_stress"]
        strong = by_slug["strong_compound_stress"]
        self.assertEqual(moderate.compound_stress.true_correlation, 0.0)
        self.assertAlmostEqual(math.tanh(moderate.config.hyperpriors.m_r), -0.5)
        self.assertEqual(strong.compound_stress.true_correlation, 0.9)
        self.assertAlmostEqual(math.tanh(strong.config.hyperpriors.m_r), -0.9)
        self.assertLess(
            math.tanh(strong.config.hyperpriors.m_r) * strong.compound_stress.true_correlation,
            0.0,
        )

    def test_anchor_study_holds_population_and_evaluation_cohort_fixed(self):
        scenarios = anchor_availability.build_scenarios(_args(anchor_availability))

        self.assertEqual(len(scenarios), 6)
        for scenario in scenarios:
            with self.subTest(anchor_count=scenario.anchor_availability.anchor_agent_count):
                counts = scenario.config.count_buckets
                anchor_count = scenario.anchor_availability.anchor_agent_count
                self.assertEqual(scenario.config.num_agents, 50)
                self.assertEqual(len(counts), 50)
                self.assertEqual(counts[:25], (1,) * 25)
                self.assertEqual(counts[25:].count(25), anchor_count)
                self.assertEqual(counts[25:].count(1), 25 - anchor_count)

    def test_anchor_summary_excludes_one_sample_context_agents(self):
        def row(agent_id: int, jeeds_sigma: float, hierarchical_sigma: float):
            return {
                "scenario_slug": "anchors_test",
                "anchor_availability_slug": "anchor_agents_000",
                "anchor_availability_label": "0 anchor agents",
                "anchor_availability_description": "test",
                "low_data_agent_count": 2,
                "low_data_observations": 1,
                "context_agent_count": 1,
                "context_low_data_agent_count": 1,
                "anchor_agent_count": 0,
                "anchor_observations": 25,
                "scenario_num_agents": 3,
                "count_buckets": "1,1,1",
                "scenario_index": 0,
                "scenario_output_dir": "/tmp/test",
                "scenario_error_plot": "/tmp/test.png",
                "seed": 10,
                "agent_id": agent_id,
                "num_observations": 1,
                "sigma_true": 1.0,
                "log_lambda_true": 0.0,
                "rationality_percent_true": 50.0,
                "jeeds_posterior_mean_sigma": jeeds_sigma,
                "jeeds_posterior_mean_log_lambda": 1.0,
                "jeeds_rationality_percent": 60.0,
                "jeeds_status": "ok",
                "hierarchical_posterior_mean_sigma": hierarchical_sigma,
                "hierarchical_posterior_mean_log_lambda": 0.5,
                "hierarchical_rationality_percent": 55.0,
                "hierarchical_status": "ok",
            }

        evaluation_rows, _overall = anchor_availability.summarize_fixed_evaluation_cohort(
            [
                row(0, 2.0, 1.5),
                row(1, 2.0, 1.5),
                row(2, 101.0, 101.0),  # one-sample context agent: must not enter the summary
            ]
        )
        jeeds_sigma = next(
            result
            for result in evaluation_rows
            if result["method"] == "jeeds" and result["metric"] == "abs_sigma_error"
        )
        self.assertEqual(jeeds_sigma["num_agents"], 2)
        self.assertAlmostEqual(jeeds_sigma["mean"], 1.0)

    def test_unified_runner_matches_reduced_scenario_counts(self):
        counts = {
            spec.slug: spec.scenario_count
            for spec in paper_runner.EXPERIMENT_SPECS
        }

        self.assertEqual(counts["agents_per_bucket"], 5)
        self.assertEqual(counts["population_shape"], 3)
        self.assertEqual(counts["decision_model"], 4)
        self.assertEqual(counts["true_correlation"], 5)
        self.assertEqual(counts["compound_stress"], 3)
        self.assertEqual(sum(counts.values()), 93)
        self.assertEqual(publication_bench.DEFAULT_ONE_D_EXPECTED_SCENARIOS, 93)

    def test_remaining_plot_inputs_rebase_to_requested_result_root(self):
        root = Path("/tmp/new_corrected_results")
        configs = remaining_plots.configs_for_results_root(root)
        self.assertTrue(configs)
        for config in configs:
            with self.subTest(experiment=config.experiment_slug):
                self.assertTrue(config.agent_level_csv.is_relative_to(root))
                self.assertTrue(config.output_stem.is_relative_to(root))


if __name__ == "__main__":
    unittest.main()
