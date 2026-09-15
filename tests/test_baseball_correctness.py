# Paper correspondence: Main `subsec:baseball`; Supplement `app:baseball_hyperpriors`/`app:baseball_sigma_gap`.
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from HJEEDS.baseball_bbip import (
    canonical_player_name,
    compute_bbip_by_pitcher,
    fetch_innings_pitched_by_name,
)
from HJEEDS.baseball_calibrate_hyperpriors import (
    COMPLETION_FILENAME,
    _validate_paper_roster_if_requested,
    agent_result_path_for,
    copy_validated_paper_hyperpriors,
    validate_calibration_completion,
    write_agent_result,
    write_calibration_outputs,
)
from HJEEDS.baseball_convergence import (
    CONVERGENCE_RUN_METADATA_FILENAME,
    CONVERGENCE_RUN_METADATA_SCHEMA_VERSION,
    finalize_convergence_run_metadata,
    restamp_convergence_run_incomplete,
    validate_complete_convergence_run_metadata,
)
from HJEEDS.baseball_hyperpriors import (
    DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH,
    build_low_confidence_hyperpriors,
    hyperprior_config_to_dict,
    load_hyperprior_config,
    load_literature_informed_baseball_hyperpriors,
    resolve_baseball_hyperpriors,
)
from HJEEDS.baseball_likelihood import (
    add_independent_log_likelihood_grids,
    compute_baseball_log_likelihood_grid,
    compute_baseball_log_likelihood_grids_by_prefix,
)
from HJEEDS.baseball_provenance import (
    EXECUTION_KERNEL_VERSION,
    PAPER_BBIP20_ROSTER_SHA256,
    PAPER_BBIP_2021_INNINGS_SHA256,
    build_baseball_run_provenance,
    roster_fingerprint,
)
from HJEEDS.baseball_pitch import (
    PitchObservation,
    StatcastAgentSpec,
    _predict_single_pitch_contexts,
    baseball_execution_skill_key,
    build_execution_displacement_distribution,
    expected_utility_with_outside_floor,
    get_agent_pitch_rows,
    sample_noisy_action,
)
from HJEEDS.baseball_separability import validate_paper_separability_inputs_and_rows
from HJEEDS.baseball_convergence_study import validate_convergence_agent_results
from HJEEDS.models import MethodEstimate, StatcastConvergenceAgentResult


class BaseballProvenanceTests(unittest.TestCase):
    @staticmethod
    def _provenance(log_lambda_grid: np.ndarray) -> dict[str, object]:
        return build_baseball_run_provenance(
            season_year=2021,
            pitch_types=("FF",),
            sigma_grid=(0.5, 1.0),
            log_lambda_grid=log_lambda_grid,
            configuration={"workflow": "provenance-test"},
        )

    def test_grid_fingerprint_ignores_one_ulp_platform_difference(self) -> None:
        login_grid = np.asarray([-6.147902198294102, 0.0, 1.0])
        compute_grid = login_grid.copy()
        compute_grid[0] = np.nextafter(compute_grid[0], np.inf)

        login = self._provenance(login_grid)
        compute = self._provenance(compute_grid)

        self.assertNotEqual(login_grid[0], compute_grid[0])
        self.assertEqual(login["log_lambda_grid"], compute["log_lambda_grid"])
        self.assertEqual(login["fingerprint_sha256"], compute["fingerprint_sha256"])

    def test_grid_fingerprint_rejects_meaningful_difference(self) -> None:
        baseline = self._provenance(np.asarray([-6.147902198294102, 0.0, 1.0]))
        changed = self._provenance(np.asarray([-6.147902198194102, 0.0, 1.0]))

        self.assertNotEqual(baseline["fingerprint_sha256"], changed["fingerprint_sha256"])


class BaseballExecutionMathTests(unittest.TestCase):
    def test_batched_single_pitch_rnn_matches_legacy_loop(self) -> None:
        import torch
        from Environments.Baseball import modelTake2

        torch.manual_seed(7)
        modelTake2.batter_indices = np.arange(5).reshape(-1, 1)
        model = modelTake2.RNN(hidden_size=32, output_size=9).eval()
        rng = np.random.default_rng(11)
        inputs = rng.normal(size=(9, 1, len(modelTake2.features))).astype(np.float32)
        inputs[:, 0, -1] = np.arange(9) % 5
        targets = (np.arange(9) % 9).reshape(-1, 1)

        legacy = _predict_single_pitch_contexts(
            model,
            inputs,
            targets,
            inference_batch_size=None,
        )
        batched = _predict_single_pitch_contexts(
            model,
            inputs,
            targets,
            inference_batch_size=4,
        )
        np.testing.assert_allclose(
            batched.detach().cpu().numpy(),
            legacy.detach().cpu().numpy(),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_full_kernel_and_outside_floor_match_direct_expectation(self) -> None:
        utility = np.asarray([[2.0, -1.0, 5.0], [4.0, 3.0, 0.5]])
        minimum = -2.5
        x_offsets = np.arange(-1, 2, dtype=float) * 0.4
        z_offsets = np.arange(-2, 3, dtype=float) * 0.4
        kernel = build_execution_displacement_distribution(
            np.asarray([[0.7**2, 0.0], [0.0, 0.45**2]]),
            0.4,
            x_offsets,
            z_offsets,
        )
        self.assertEqual(kernel.shape, (3, 5))
        self.assertLessEqual(float(kernel.sum()), 1.0 + 1e-6)

        actual = expected_utility_with_outside_floor(utility, kernel, minimum)
        direct = np.empty_like(utility)
        center = (utility.shape[0] - 1, utility.shape[1] - 1)
        for intended_x in range(utility.shape[0]):
            for intended_z in range(utility.shape[1]):
                valid_mass = 0.0
                weighted_utility = 0.0
                for executed_x in range(utility.shape[0]):
                    for executed_z in range(utility.shape[1]):
                        probability = kernel[
                            center[0] + executed_x - intended_x,
                            center[1] + executed_z - intended_z,
                        ]
                        valid_mass += probability
                        weighted_utility += probability * utility[executed_x, executed_z]
                direct[intended_x, intended_z] = (
                    weighted_utility + (1.0 - valid_mass) * minimum
                )
        np.testing.assert_allclose(actual, direct, rtol=1e-12, atol=1e-12)

    def test_noise_stream_advances_and_is_reproducible(self) -> None:
        first_rng = np.random.default_rng(123)
        first_sequence = [sample_noisy_action(first_rng, (0.0, 0.0), 1.0) for _ in range(3)]
        second_rng = np.random.default_rng(123)
        second_sequence = [sample_noisy_action(second_rng, (0.0, 0.0), 1.0) for _ in range(3)]
        self.assertEqual(first_sequence, second_sequence)
        self.assertEqual(len(set(first_sequence)), 3)

    def test_streamed_likelihood_accumulation_equals_batch(self) -> None:
        sigma_grid = np.asarray([0.5, 1.0])
        log_lambda_grid = np.log(np.asarray([0.2, 1.0, 3.0]))
        targets = np.asarray([[-0.25, 0.0], [0.25, 0.0], [0.0, 0.5]])
        all_covs = {
            baseball_execution_skill_key(float(sigma)): np.eye(2) * float(sigma) ** 2
            for sigma in sigma_grid
        }
        observations = []
        for index, action in enumerate(((-0.1, 0.1), (0.2, 0.3), (0.0, 0.45))):
            observations.append(
                PitchObservation(
                    executed_action=action,
                    observed_reward=0.0,
                    evs_per_execution_skill={
                        baseball_execution_skill_key(float(sigma)): np.asarray(
                            [0.5 + index, -0.2 * sigma, 1.0 - 0.1 * index]
                        )
                        for sigma in sigma_grid
                    },
                    min_utility=-1.0,
                )
            )
        kwargs = dict(
            possible_targets_feet=targets,
            all_covs=all_covs,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
            delta=0.1,
        )
        batch = compute_baseball_log_likelihood_grid(
            pitch_observations=observations,
            **kwargs,
        )
        streamed = None
        for observation in observations:
            streamed = add_independent_log_likelihood_grids(
                streamed,
                compute_baseball_log_likelihood_grid(
                    pitch_observations=[observation],
                    **kwargs,
                ),
            )
        np.testing.assert_array_equal(streamed, batch)

        prefixes = compute_baseball_log_likelihood_grids_by_prefix(
            pitch_observations=observations,
            prefix_lengths=(1, 2, 3),
            **kwargs,
        )
        for prefix_length in (1, 2, 3):
            repeated = compute_baseball_log_likelihood_grid(
                pitch_observations=observations[:prefix_length],
                **kwargs,
            )
            np.testing.assert_array_equal(prefixes[prefix_length], repeated)


class BaseballDataAndRosterTests(unittest.TestCase):
    def test_pitch_order_has_deterministic_within_day_tie_breaks(self) -> None:
        frame = pd.DataFrame(
            {
                "pitcher": [7] * 5,
                "pitch_type": ["FF"] * 5,
                "game_date": ["2021-05-01", "2021-05-02", "2021-05-02", "2021-05-02", "2021-05-02"],
                "game_pk": [1, 2, 2, 2, 3],
                "at_bat_number": [1, 1, 2, 2, 1],
                "pitch_number": [1, 1, 1, 2, 1],
            }
        )
        ordered = get_agent_pitch_rows(frame, 7, "FF")
        keys = list(
            ordered[["game_date", "game_pk", "at_bat_number", "pitch_number"]]
            .itertuples(index=False, name=None)
        )
        self.assertEqual(
            keys,
            [
                ("2021-05-02", 3, 1, 1),
                ("2021-05-02", 2, 2, 2),
                ("2021-05-02", 2, 2, 1),
                ("2021-05-02", 2, 1, 1),
                ("2021-05-01", 1, 1, 1),
            ],
        )

    def test_name_keys_repair_all_known_cache_variants(self) -> None:
        pairs = (
            ("Bukauskas, J.B.", "bukauskas jb"),
            ("De Jong, Chase", "jong chase de"),
            ("Hammer, JD", "hammer j.d."),
            ("De Los Santos, Enyel", "santos enyel de los"),
            ("Ponce de Leon, Daniel", "leon daniel ponce de"),
            ("De León, José", r"le\xc3\xb3n jos\xc3\xa9 de"),
        )
        for statcast_name, cache_name in pairs:
            self.assertEqual(canonical_player_name(statcast_name), canonical_player_name(cache_name))

    def test_bbip_filters_season_and_keeps_zero_walk_pitchers(self) -> None:
        frame = pd.DataFrame(
            {
                "game_year": [2021, 2021, 2022, 2022],
                "pitcher": [1, 2, 1, 2],
                "player_name": ["Alpha, Ann", "Beta, Bob", "Alpha, Ann", "Beta, Bob"],
                "events": ["field_out", "walk", "walk", "walk"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "bbip_innings_cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "season_year": 2021,
                        "source": "test",
                        "innings_by_name": {"alpha ann": 10.0, "beta bob": 20.0},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"BBIP_CACHE_PATH": str(cache)}):
                table = compute_bbip_by_pitcher(frame, season_year=2021)
        by_id = table.set_index("pitcher_id")
        self.assertEqual(int(by_id.loc[1, "walks"]), 0)
        self.assertEqual(float(by_id.loc[1, "bbip"]), 0.0)
        self.assertEqual(int(by_id.loc[2, "walks"]), 1)

    def test_bundled_innings_artifact_is_preferred_to_live_fetches(self) -> None:
        with mock.patch(
            "HJEEDS.baseball_bbip._fetch_innings_from_bref",
            side_effect=AssertionError("live BRef fetch must not run"),
        ), mock.patch(
            "HJEEDS.baseball_bbip._fetch_innings_from_fangraphs",
            side_effect=AssertionError("live FanGraphs fetch must not run"),
        ):
            innings, source = fetch_innings_pitched_by_name(2021)
        self.assertGreater(len(innings), 500)
        self.assertTrue(source.startswith("bundled:baseball_innings_pitched_2021.json"))

    def test_bbip_proxy_counts_all_retained_requested_year_walk_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "game_year": [2021, 2021, 2021, 2022],
                "game_date": ["2021-03-15", "2021-11-02", "2021-07-01", "2022-05-01"],
                "pitcher": [1, 1, 1, 1],
                "player_name": ["Alpha, Ann"] * 4,
                "events": ["walk", "walk", "field_out", "walk"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "bbip_innings_cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "season_year": 2021,
                        "source": "test",
                        "innings_by_name": {"alpha ann": 10.0},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"BBIP_CACHE_PATH": str(cache)}):
                table = compute_bbip_by_pitcher(frame, season_year=2021)
        self.assertEqual(int(table.iloc[0]["walks"]), 2)
        self.assertAlmostEqual(float(table.iloc[0]["bbip"]), 0.2)


class BaseballPublicationGuardTests(unittest.TestCase):
    @staticmethod
    def _estimate(method: str) -> MethodEstimate:
        return MethodEstimate(
            method_name=method,
            posterior_mean_sigma=1.0,
            posterior_mean_log_lambda=2.0,
            status="ok",
        )

    def test_convergence_result_validation_rejects_failed_status(self) -> None:
        valid = StatcastConvergenceAgentResult(
            seed=1,
            agent_id=0,
            pitcher_id=10,
            pitch_type="FF",
            convergence_n=5,
            num_observations=5,
            num_reference_observations=5,
            reference=self._estimate("reference"),
            jeeds=self._estimate("jeeds"),
            hierarchical=self._estimate("hierarchical"),
            abs_sigma_drift_vs_full_jeeds=0.0,
            abs_log_lambda_drift_vs_full_jeeds=0.0,
            abs_sigma_drift_vs_full_hierarchical=0.0,
            abs_log_lambda_drift_vs_full_hierarchical=0.0,
        )
        validate_convergence_agent_results([valid])
        invalid = StatcastConvergenceAgentResult(
            **{
                **valid.__dict__,
                "hierarchical": MethodEstimate(method_name="hierarchical", status="failed"),
            }
        )
        with self.assertRaisesRegex(ValueError, "H-JEEDS status"):
            validate_convergence_agent_results([invalid])

        negative_sigma = StatcastConvergenceAgentResult(
            **{
                **valid.__dict__,
                "jeeds": MethodEstimate(
                    method_name="jeeds",
                    posterior_mean_sigma=0.0,
                    posterior_mean_log_lambda=2.0,
                    status="ok",
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_convergence_agent_results([negative_sigma])

        negative_drift = StatcastConvergenceAgentResult(
            **{**valid.__dict__, "abs_sigma_drift_vs_full_jeeds": -0.01}
        )
        with self.assertRaisesRegex(ValueError, "jeeds_sigma_drift"):
            validate_convergence_agent_results([negative_drift])

        with self.assertRaisesRegex(ValueError, r"expected \[5, 10\]"):
            validate_convergence_agent_results(
                [valid],
                expected_agent_ids={0},
                expected_checkpoints={5, 10},
            )

    def test_separability_validation_requires_complete_cohort(self) -> None:
        tiers = {
            10: {"tier": "bottom", "bbip": 0.1},
            20: {"tier": "top", "bbip": 0.8},
        }
        agent_rows = [
            {
                "pitcher_id": pitcher,
                "convergence_n": 5,
                "jeeds_status": "ok",
                "hierarchical_status": "ok",
            }
            for pitcher in (10, 20)
        ]
        summaries = [
            {
                "method": method,
                "metric": metric,
                "convergence_n": 5,
                "num_bottom": 1,
                "num_top": 1,
                "mean_bottom": 1.0,
                "mean_top": 2.0,
                "mean_gap_top_minus_bottom": 1.0,
                "auc": 1.0,
            }
            for method in ("jeeds", "hierarchical")
            for metric in ("sigma", "log_lambda")
        ]
        validate_paper_separability_inputs_and_rows(agent_rows, tiers, summaries)
        with self.assertRaisesRegex(ValueError, "checkpoint 5"):
            validate_paper_separability_inputs_and_rows(agent_rows[:1], tiers, summaries)

    def test_stale_hyperprior_file_fails_closed(self) -> None:
        payload = hyperprior_config_to_dict(build_low_confidence_hyperpriors())
        payload["_provenance"] = {"execution_kernel_version": "legacy-v1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "stale or incomplete MLB provenance"):
                load_hyperprior_config(path, require_current_execution_kernel=True)

    def test_committed_paper_hyperprior_is_current_or_explicitly_stale(self) -> None:
        try:
            resolve_baseball_hyperpriors(preset="baseball-2021-ff")
        except RuntimeError:
            from HJEEDS.baseball_hyperpriors import DEFAULT_BASEBALL_HYPERPRIORS_2021_FF_PATH

            payload = json.loads(
                DEFAULT_BASEBALL_HYPERPRIORS_2021_FF_PATH.read_text(encoding="utf-8")
            )
            provenance = payload.get("_provenance") or {}
            self.assertNotEqual(provenance.get("execution_kernel_version"), EXECUTION_KERNEL_VERSION)

    def test_literature_informed_paper_hyperprior_is_frozen_and_current(self) -> None:
        config = load_literature_informed_baseball_hyperpriors()
        self.assertEqual(config.mean_vector, (math.log(0.5), math.log(100.0)))
        self.assertEqual(config.covariance_diagonal, (2.25, 16.0))
        self.assertEqual(config.log_tau_eta_mean, math.log(0.35))
        self.assertEqual(config.log_tau_rho_mean, 0.0)
        self.assertEqual(config.m_r, 0.0)
        self.assertEqual(
            resolve_baseball_hyperpriors(preset="baseball-literature-informed"),
            config,
        )
        payload = json.loads(
            DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH.read_text(encoding="utf-8")
        )
        provenance = payload["_provenance"]
        self.assertEqual(provenance["execution_kernel_version"], EXECUTION_KERNEL_VERSION)
        self.assertEqual(provenance["status"], "fixed-before-corrected-MLB-evaluation")


class BaseballCalibrationCompletionTests(unittest.TestCase):
    @staticmethod
    def _write_fixture(output_dir: Path) -> None:
        roster_rows = [
            {"agent_id": 0, "pitcher_id": 10, "pitch_type": "FF"},
            {"agent_id": 1, "pitcher_id": 20, "pitch_type": "FF"},
        ]
        roster_hash = roster_fingerprint(
            [
                (row["agent_id"], row["pitcher_id"], row["pitch_type"])
                for row in roster_rows
            ]
        )
        run_provenance = build_baseball_run_provenance(
            season_year=2021,
            pitch_types=("FF",),
            sigma_grid=(0.5, 1.0),
            log_lambda_grid=(0.0, 1.0),
            configuration={
                "workflow": "baseball-hyperprior-calibration",
                "min_pitches_per_agent": 1,
                "max_pitches_per_agent": 1,
                "max_agents": 2,
                "confidence": "low",
                "roster_selector": {
                    "all_eligible_agents": False,
                    "pitcher_ids": [10, 20],
                    "top_pitchers": None,
                    "bbip_extremes": None,
                },
                "roster_fingerprint": roster_hash,
                "num_agents": 2,
            },
        )
        output_dir.mkdir(parents=True)
        (output_dir / "calibration_roster.json").write_text(
            json.dumps(roster_rows),
            encoding="utf-8",
        )
        (output_dir / "calibration_roster_metadata.json").write_text(
            json.dumps(
                {
                    "season_year": 2021,
                    "pitch_types": ["FF"],
                    "num_agents": 2,
                    "run_provenance": run_provenance,
                }
            ),
            encoding="utf-8",
        )
        rows = [
            {
                "agent_id": index,
                "pitcher_id": pitcher_id,
                "pitch_type": "FF",
                "num_observations": 1,
                "posterior_mean_sigma": 1.0 + 0.1 * index,
                "posterior_mean_log_lambda": 2.0 + 0.1 * index,
                "map_sigma": 1.0,
                "map_log_lambda": 2.0,
                "status": "ok",
                "notes": "",
                "provenance_fingerprint": run_provenance["fingerprint_sha256"],
            }
            for index, pitcher_id in enumerate((10, 20))
        ]
        for row in rows:
            write_agent_result(
                agent_result_path_for(output_dir, int(row["agent_id"])),
                row,
            )
        summary = {
            "season_year": 2021,
            "pitch_types": ["FF"],
            "num_agents": 2,
            "num_completed_agent_results": 2,
            "num_missing_agent_results": 0,
            "missing_agent_ids": [],
            "num_successful_estimates": 2,
            "confidence": "low",
            "roster_selector": {
                "all_eligible_agents": False,
                "pitcher_ids": [10, 20],
                "top_pitchers": None,
                "bbip_extremes": None,
            },
            "min_pitches_per_agent": 1,
            "max_pitches_per_agent": 1,
            "max_agents": 2,
            "run_provenance": run_provenance,
            "skill_grid": {"test": True},
        }
        write_calibration_outputs(
            output_dir,
            {
                "rows": rows,
                "summary": summary,
                "hyperpriors": build_low_confidence_hyperpriors(),
            },
        )

    def test_completion_hashes_outputs_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "calibration"
            self._write_fixture(output_dir)
            completion = validate_calibration_completion(output_dir)
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(completion["num_completed_agent_results"], 2)
            self.assertFalse(completion["publication_ready_2021_ff"])

            csv_path = output_dir / "jeeds_calibration_agent_estimates.csv"
            csv_path.write_text(
                csv_path.read_text(encoding="utf-8") + "tampered\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_calibration_completion(output_dir)

    def test_canonical_design_rejects_a_different_528_agent_roster(self) -> None:
        fake_roster = tuple(
            StatcastAgentSpec(agent_id=index, pitcher_id=100_000 + index, pitch_type="FF")
            for index in range(528)
        )
        with self.assertRaisesRegex(RuntimeError, "Canonical paper calibration roster mismatch"):
            _validate_paper_roster_if_requested(
                fake_roster,
                season_year=2021,
                pitch_types=("FF",),
                min_pitches_per_agent=100,
                max_pitches_per_agent=None,
                max_agents=None,
                confidence="low",
                roster_selector={
                    "all_eligible_agents": True,
                    "pitcher_ids": None,
                    "top_pitchers": None,
                    "bbip_extremes": None,
                },
            )

    def test_failed_output_commit_leaves_incomplete_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "calibration"
            with mock.patch(
                "HJEEDS.baseball_calibrate_hyperpriors.write_hyperprior_config",
                side_effect=RuntimeError("injected write failure"),
            ), self.assertRaisesRegex(RuntimeError, "injected write failure"):
                self._write_fixture(output_dir)
            completion = json.loads(
                (output_dir / COMPLETION_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(completion["status"], "incomplete")
            self.assertEqual(completion["stage"], "committing-outputs")

    def test_completion_detects_tampered_agent_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "calibration"
            self._write_fixture(output_dir)
            agent_path = agent_result_path_for(output_dir, 0)
            agent_path.write_text(
                agent_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "agent output hash mismatch"):
                validate_calibration_completion(output_dir)

    def test_noncanonical_completion_cannot_be_copied_as_paper_prior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "calibration"
            self._write_fixture(output_dir)
            with self.assertRaisesRegex(RuntimeError, "not the canonical complete 528/528"):
                copy_validated_paper_hyperpriors(
                    output_dir,
                    Path(directory) / "paper_hyperpriors.json",
                )

    def test_validated_copy_is_atomic_and_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "calibration"
            output_dir.mkdir()
            source = output_dir / "suggested_hyperpriors.json"
            source.write_bytes(b'{"validated": true}\n')
            expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            destination = Path(directory) / "installed.json"
            destination.write_bytes(b"old")
            with mock.patch(
                "HJEEDS.baseball_calibrate_hyperpriors.validate_calibration_completion",
                return_value={"output_sha256": {source.name: expected_hash}},
            ):
                copied_hash = copy_validated_paper_hyperpriors(output_dir, destination)
            self.assertEqual(copied_hash, expected_hash)
            self.assertEqual(destination.read_bytes(), source.read_bytes())


class BaseballConvergenceCompletionTests(unittest.TestCase):
    _RESULT_FILENAMES = (
        "convergence_agent_level_results.csv",
        "summary_by_N.csv",
        "summary_overall.csv",
        "drift_by_N.png",
        "drift_by_N_proportional.png",
        "population_fit_diagnostics.json",
    )

    @classmethod
    def _write_incomplete_fixture(cls, output_dir: Path) -> None:
        output_dir.mkdir(parents=True)
        for index, name in enumerate(cls._RESULT_FILENAMES):
            (output_dir / name).write_bytes(f"artifact-{index}\n".encode("utf-8"))
        payload = {
            "schema_version": CONVERGENCE_RUN_METADATA_SCHEMA_VERSION,
            "completion_status": "incomplete",
            "run_provenance": {},
            "hyperprior": {},
            "cohort": {"frozen_roster": False},
            "nonpaper_overrides": {
                "allow_partial_caches": False,
                "allow_separability_failure": False,
            },
            "artifact_sha256": {},
            "artifact_sizes_bytes": {},
        }
        (output_dir / CONVERGENCE_RUN_METADATA_FILENAME).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_completion_hashes_outputs_and_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "convergence"
            self._write_incomplete_fixture(output_dir)
            finalize_convergence_run_metadata(output_dir)
            completion = validate_complete_convergence_run_metadata(output_dir)
            self.assertEqual(completion["completion_status"], "complete")
            self.assertEqual(set(completion["artifact_sha256"]), set(self._RESULT_FILENAMES))

            summary_path = output_dir / "summary_by_N.csv"
            summary_path.write_bytes(summary_path.read_bytes() + b"tampered\n")
            with self.assertRaisesRegex(ValueError, "size changed|hash changed"):
                validate_complete_convergence_run_metadata(output_dir)

    def test_plot_rewrite_restamp_invalidates_bundle_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "convergence"
            self._write_incomplete_fixture(output_dir)
            finalize_convergence_run_metadata(output_dir)
            restamp_convergence_run_incomplete(output_dir)
            payload = json.loads(
                (output_dir / CONVERGENCE_RUN_METADATA_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["completion_status"], "incomplete")
            self.assertEqual(payload["artifact_sha256"], {})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_complete_convergence_run_metadata(output_dir)

    def test_paper_roster_and_innings_constants_match_tracked_inputs(self) -> None:
        pitcher_ids = (
            521230,
            458708,
            455119,
            605154,
            594798,
            676083,
            621107,
            552640,
            571948,
            433589,
            656686,
            669135,
            607457,
            476594,
            643778,
            656793,
            663399,
            663366,
            665620,
            660600,
        )
        actual_roster_hash = roster_fingerprint(
            [(index, pitcher_id, "FF") for index, pitcher_id in enumerate(pitcher_ids)]
        )
        self.assertEqual(actual_roster_hash, PAPER_BBIP20_ROSTER_SHA256)

        innings_path = (
            Path(__file__).resolve().parent.parent
            / "HJEEDS"
            / "data"
            / "baseball_innings_pitched_2021.json"
        )
        actual_innings_hash = hashlib.sha256(innings_path.read_bytes()).hexdigest()
        self.assertEqual(actual_innings_hash, PAPER_BBIP_2021_INNINGS_SHA256)


if __name__ == "__main__":
    unittest.main()
