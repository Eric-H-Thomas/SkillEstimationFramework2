"""Run JEEDS on Blackhawks shot data to estimate player execution skill.

This module stitches together the Snowflake-backed Blackhawks API with the
framework's JEEDS implementation (``JointMethodQRE``).  The workflow:

1. Pull per-shot metadata for a given player across selected games via
   :func:`BlackhawksAPI.query_player_game_info`.
2. Fetch precomputed reward surfaces (post-shot xG probability grids) for each
   shot via :func:`BlackhawksAPI.get_game_shot_maps`.
3. Convert those raw SQL rows and reward surfaces into angular heatmaps and
   covariances expected by the JEEDS estimator for the hockey domain.
4. Feed each observed shot into JEEDS and return the final MAP execution-skill
   and rationality estimates.

This integration uses the Blackhawks analytics' precomputed shot maps, which
incorporate detailed models of shooting position, angle, goalie positioning,
and other factors to estimate expected value for each possible aim direction.

Example
-------
from BlackhawksSkillEstimation.BlackhawksJEEDS import estimate_player_skill
estimates = estimate_player_skill(player_id=950160, game_ids=[44604, 270247])
print(f"Execution skill: {estimates['execution_skill']:.4f} rad (lower=better)")
print(f"Rationality: {estimates['rationality']:.2f} (higher=better, experimental)")
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from BlackhawksAPI import query_player_game_info, get_game_shot_maps
from Environments.Hockey import getAngularHeatmapsPerPlayer as angular_heatmaps
from Estimators.joint import JointMethodQRE


# Default number of execution skill hypotheses for JEEDS estimation.
# Trade-off between resolution and computational cost:
#   10 hypotheses: ~0.08 rad spacing, fast, good for testing
#   20 hypotheses: ~0.04 rad spacing, balanced for most uses
#   30 hypotheses: ~0.025 rad spacing, matches production experiments
#   50 hypotheses: ~0.015 rad spacing, high precision for research
DEFAULT_NUM_EXECUTION_SKILLS = 30  # Recommended for overnight cluster runs


@dataclass
class SimpleHockeySpaces:
    """Minimal space object exposing the hooks JEEDS expects for hockey.

    The production framework builds a rich :class:`SpacesHockey` structure that
    caches precomputed EV surfaces, covariances, and target grids.  For this
    integration we only need a subset of those attributes:

    ``possibleTargets``
        Flattened list of (y, z) candidates used when evaluating shot PDFs.
    ``delta``
        Grid spacing (dy, dz); maintained for parity with the framework API.
    ``allCovs``
        Covariance matrices keyed by ``getKey([x, x], 0.0)``.
    """

    y_grid: np.ndarray
    z_grid: np.ndarray
    candidate_execution_skills: Sequence[float]
    possible_targets: np.ndarray | None = None
    possibleTargets: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        dy = np.diff(self.y_grid).mean() if len(self.y_grid) > 1 else 1.0
        dz = np.diff(self.z_grid).mean() if len(self.z_grid) > 1 else 1.0
        self.delta = (float(dy), float(dz))

        if self.possible_targets is None:
            self.possibleTargets = np.array(
                [(y, z) for y in self.y_grid for z in self.z_grid], dtype=float
            )
        else:
            self.possibleTargets = np.array(self.possible_targets, dtype=float)

        self.all_covs: dict[str, np.ndarray] = {}
        for skill in self.candidate_execution_skills:
            key = self.get_key([skill, skill], r=0.0)
            self.all_covs[key] = self._covariance_from_skill(skill)

    # Compare to getCovMatrix in hockey.py. Assume rho=0, so we simplify to np.diag
    @staticmethod
    def _covariance_from_skill(skill: float) -> np.ndarray:
        """Return a 2D diagonal covariance matrix for execution error in angular space.
        
        The input skill is the standard deviation (in radians) of execution error
        in both direction and elevation. Variance = skill², so:
        - Higher skill → larger variance → worse execution (looser spread)
        - Lower skill → smaller variance → better execution (tighter cluster)
        """
        variance = max(skill, 1e-6) ** 2
        return np.diag([variance, variance])

    '''
    Examples of how skill (in radians) translates to variance:
    skill: 0.004; variance: 0.000016 (elite execution, tight cluster)
    skill: 0.02;  variance: 0.0004   (strong execution, tight spread)
    skill: 0.1;   variance: 0.01     (moderate execution)
    skill: 0.3;   variance: 0.09     (loose execution, wide spread)
    skill: 0.6;   variance: 0.36     (poor execution, very wide spread)
    skill: 0.785 (π/4); variance: 0.616 (terrible execution, nearly random)
    '''

    @staticmethod
    def get_key(info: Sequence[float], r: float) -> str:
        return "|".join(map(str, info)) + f"|{r}"


@dataclass
class JEEDSInputs:
    spaces_per_shot: list[SimpleHockeySpaces]
    actions: list[list[float]]
    info_rows: list[dict[str, object]]


# ============================================================================
# ARCHITECTURAL NOTE: Spaces as Reward Maps/Reward Surfaces
# ============================================================================
#
# A "Space" in this framework is a REWARD MAP—a lookup table that tells the
# estimator: "For this shot location, here are the possible aims and how
# valuable each one is."
#
# **SpacesHockey (production framework):**
#   - Pre-computes reward surfaces for EVERY POSSIBLE SHOT LOCATION
#   - Uses expensive scipy.signal.convolve2d operations with skill-based kernels
#   - Stores all results in memory or on disk
#   - Reuses the same Space object across many shots in an experiment
#   - Used by: Estimators when processing simulation data
#
# **SimpleHockeySpaces (this module):**
#   - Uses Blackhawks precomputed reward surfaces (value_map) for each shot
#   - Retrieved from hawks_analytics.post_shot_xg_value_maps via get_game_shot_maps()
#   - Each shot's value_map is a 120x72 grid of post-shot xG probabilities
#   - Creates a new Space instance per shot, discards it after use
#   - No expensive local precomputation needed; data comes from Snowflake
#   - Used by: BlackhawksJEEDS when processing real player data from Snowflake
#
# Both expose the same interface (possibleTargets, all_covs, delta, get_key())
# so JointMethodQRE treats them identically during skill estimation.


def transform_shots_for_jeeds(
    df,
    shot_maps: dict[int, dict[str, object]],
    candidate_skills: Sequence[float],
) -> JEEDSInputs:
    """Convert SQL rows into the structures JEEDS expects.

    Parameters
    ----------
    df:
        DataFrame returned by :func:`query_player_game_info`.
    shot_maps:
        Dictionary of shot metadata keyed by event_id_hawks, returned by
        :func:`get_game_shot_maps`. Contains 'value_map', 'net_cov', and
        'net_coords' for each shot.
    candidate_skills:
        Execution-skill hypotheses passed to JEEDS.
    """

    df = df.rename(columns=str.lower)

    info_rows: list[dict[str, object]] = []
    actions: list[list[float]] = []
    spaces_per_shot: list[SimpleHockeySpaces] = []

    for _, row in df.iterrows():
        event_id = int(row["event_id"])
        
        # Skip shots without precomputed Blackhawks reward surfaces
        if event_id not in shot_maps:
            continue
        
        shot_map_data = shot_maps[event_id]
        base_ev = shot_map_data["value_map"]
        
        # Resize xG map from 72x120 (queries.py) to 60x40 (expected by getAngularHeatmap)
        # The zoom factors are: 60/72 ≈ 0.833 for Y, 40/120 ≈ 0.333 for Z
        base_ev = zoom(base_ev, (60 / base_ev.shape[0], 40 / base_ev.shape[1]), order=1)

        player_location = np.array([float(row["start_x"]), float(row["start_y"])])
        executed_action = np.array([float(row["location_y"]), float(row["location_z"])])

        # Convert Blackhawks Cartesian reward surface to anglular coordinates
        (
            dirs,
            elevations,
            _,
            grid_targets_angular,
            _,
            _,
            _,
            grid_utilities_computed,
            executed_action_angular,
            skip,
            _,
        ) = angular_heatmaps.getAngularHeatmap(
            base_ev,
            player_location,
            executed_action,
        )

        if skip:
            continue

        spaces = SimpleHockeySpaces(
            np.array(dirs),
            np.array(elevations),
            candidate_skills,
            possible_targets=grid_targets_angular.reshape(-1, 2),
        )

        entry = {"evsPerXskill": {}, "maxEVPerXskill": {}, "focalActions": []}

        # Compute skill-dependent EV surface smoothing.
        # Lower xskill (tight execution) → sharp EV surface (player hits what they aim at)
        # Higher xskill (loose execution) → blurred EV surface (expected value smears across potential hit locations)
        # NOTE: We generate a blurred version of the xG surface for each xskill hypothesis.
        dir_bin_size = (dirs[-1] - dirs[0]) / (len(dirs) - 1) if len(dirs) > 1 else 0.01
        elev_bin_size = (elevations[-1] - elevations[0]) / (len(elevations) - 1) if len(elevations) > 1 else 0.01
        avg_bin_size = (dir_bin_size + elev_bin_size) / 2

        for skill in candidate_skills:
            key = spaces.get_key([skill, skill], r=0.0)
            # sigma in grid bins: skill (rad) / avg_bin_size (rad/bin) = bins
            # Clamp to avoid extreme smoothing or no smoothing.
            sigma = max(min(skill / avg_bin_size, 1.0), 1e-3)
            evs = gaussian_filter(grid_utilities_computed, sigma=sigma)

            entry["evsPerXskill"][key] = evs
            entry["maxEVPerXskill"][key] = float(np.max(evs))

            # Find most rational target location for this xskill's blurred xG surface
            best_idx = int(np.argmax(evs))
            iy, iz = np.unravel_index(best_idx, (len(dirs), len(elevations)))
            entry["focalActions"].append(
                [float(grid_targets_angular[iy, iz, 0]), float(grid_targets_angular[iy, iz, 1])]
            )

        info_rows.append(entry)
        actions.append([float(executed_action_angular[0]), float(executed_action_angular[1])])
        spaces_per_shot.append(spaces)

    return JEEDSInputs(spaces_per_shot=spaces_per_shot, actions=actions, info_rows=info_rows)


def ensure_results_directories(results_folder: Path) -> None:
    times_dir = results_folder / "times" / "estimators"
    times_dir.mkdir(parents=True, exist_ok=True)


def estimate_player_skill(
    player_id: int,
    game_ids: Sequence[int],
    *,
    candidate_skills: Sequence[float] | None = None,
    num_planning_skills: int = 25,
    results_folder: Path | str = Path("blackhawks-jeeds"),
    rng_seed: int | None = 0,
    return_intermediate_estimates: bool = False,
) -> dict[str, object]:
    """Return JEEDS MAP estimates of execution skill and rationality for a player.

    Parameters mirror :func:`transform_shots_for_jeeds` with additional JEEDS
    configuration knobs.  The estimator relies on the standard Blackhawks
    Snowflake environment variables to authenticate when fetching shots.

    Parameters
    ----------
    return_intermediate_estimates : bool
        If True, return a 'skill_log' key with MAP estimates after each shot.
        Default is False.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'execution_skill': MAP estimate of xskill in radians.
          **Lower is better** (tight execution). Range: [0.004, π/4].
        - 'rationality': MAP estimate of decision-making optimality.
          **Higher is better** (optimal aim selection).
          **EXPERIMENTAL**: May not fully account for game context (defenders, time pressure).
        - 'num_shots': Number of shots used in estimation.
        - 'skill_log' (if return_intermediate_estimates=True): List of dicts with
          intermediate MAP estimates after each shot, including 'shot_idx',
          'execution_skill', and 'rationality'.
    """

    # Default to DEFAULT_NUM_EXECUTION_SKILLS execution skill hypotheses spanning
    # practical angular range (radians). 0.004 rad ≈ 0.23°, π/4 ≈ 45°: the same
    # range used in getAngularHeatMapsPerPlayer.py for min_skill and max_skill.
    # Adjust DEFAULT_NUM_EXECUTION_SKILLS at top of file for accuracy vs. speed trade-off.
    candidate_skills = list(
        candidate_skills or np.linspace(0.004, np.pi / 4, DEFAULT_NUM_EXECUTION_SKILLS)
    )

    df = query_player_game_info(player_id=player_id, game_ids=list(game_ids))
    if df.empty:
        raise RuntimeError("No shot data returned for the requested player/games.")

    # Fetch Blackhawks precomputed reward surfaces for each game
    shot_maps: dict[int, dict[str, object]] = {}
    for game_id in game_ids:
        try:
            game_shot_maps = get_game_shot_maps(game_id)
            shot_maps.update(game_shot_maps)
        except Exception as e:
            # Log but continue if a game's shot maps aren't available
            print(f"Warning: Could not fetch shot maps for game {game_id}: {e}")

    jeeds_inputs = transform_shots_for_jeeds(
        df,
        shot_maps=shot_maps,
        candidate_skills=candidate_skills,
    )
    if not jeeds_inputs.actions:
        raise RuntimeError("No usable shot data remained after angular conversion.")

    estimator = JointMethodQRE(list(candidate_skills), num_planning_skills, "hockey-multi")
    results_folder = Path(results_folder)
    ensure_results_directories(results_folder)

    rng = np.random.default_rng(rng_seed)
    tag = f"Player_{player_id}"

    skill_log: list[dict[str, object]] = []
    xskill_key = f"{estimator.method_type}-MAP-{estimator.num_execution_skills}-{estimator.num_rationality_levels}-xSkills"
    pskill_key = f"{estimator.method_type}-MAP-{estimator.num_execution_skills}-{estimator.num_rationality_levels}-pSkills"

    for idx, (spaces, info, action) in enumerate(
        zip(jeeds_inputs.spaces_per_shot, jeeds_inputs.info_rows, jeeds_inputs.actions)
    ):
        estimator.add_observation(
            rng,
            spaces,
            state=None,
            action=action,
            resultsFolder=str(results_folder),
            tag=tag,
            infoPerRow=info,
            s=str(idx),
        )

        # Optionally track intermediate MAP estimates after each shot
        if return_intermediate_estimates:
            results = estimator.get_results()
            xskill_estimates = results.get(xskill_key, [])
            pskill_estimates = results.get(pskill_key, [])
            
            if xskill_estimates and pskill_estimates:
                skill_log.append({
                    "shot_idx": idx,
                    "execution_skill": float(xskill_estimates[-1]),
                    "rationality": float(pskill_estimates[-1]),
                })

    results = estimator.get_results()
    
    # Extract MAP estimates for both execution skill and rationality
    xskill_estimates = results.get(xskill_key, [])
    pskill_estimates = results.get(pskill_key, [])
    
    if not xskill_estimates:
        raise RuntimeError("JEEDS returned no MAP execution-skill estimate.")
    if not pskill_estimates:
        raise RuntimeError("JEEDS returned no MAP rationality estimate.")

    result = {
        "execution_skill": float(xskill_estimates[-1]),
        "rationality": float(pskill_estimates[-1]),
        "num_shots": len(jeeds_inputs.actions),
    }
    
    if return_intermediate_estimates:
        result["skill_log"] = skill_log

    return result


def estimate_multiple_players(
    player_ids: Sequence[int],
    game_ids: Sequence[int],
    *,
    candidate_skills: Sequence[float] | None = None,
    num_planning_skills: int = 25,
    results_folder: Path | str = Path("blackhawks-jeeds"),
    rng_seed: int | None = 0,
    capture_skill_logs: bool = False,
) -> list[dict[str, object]]:
    """Estimate execution skill and rationality for multiple players.

    Parameters
    ----------
    player_ids : Sequence[int]
        List of player identifiers to estimate.
    game_ids : Sequence[int]
        List of game identifiers to include in estimation for all players.
    capture_skill_logs : bool
        If True, capture intermediate MAP estimates after each shot for each player.
        Default is False.
    
    Other parameters are passed to :func:`estimate_player_skill`.

    Returns
    -------
    list[dict]
        List of result dictionaries, one per player, each containing:
        - 'player_id': The player identifier.
        - 'execution_skill': MAP estimate of xskill in radians.
        - 'rationality': MAP estimate of decision-making optimality.
        - 'num_shots': Number of shots used in estimation.
        - 'skill_log' (if capture_skill_logs=True): Intermediate estimates per shot.
        - 'status': 'success' or 'error'.
        - 'error' (if status='error'): Error message.
    """
    results: list[dict[str, object]] = []

    for player_id in player_ids:
        try:
            player_result = estimate_player_skill(
                player_id=player_id,
                game_ids=game_ids,
                candidate_skills=candidate_skills,
                num_planning_skills=num_planning_skills,
                results_folder=results_folder,
                rng_seed=rng_seed,
                return_intermediate_estimates=capture_skill_logs,
            )
            player_result["player_id"] = player_id
            player_result["status"] = "success"
            results.append(player_result)
        except Exception as e:
            results.append({
                "player_id": player_id,
                "status": "error",
                "error": str(e),
            })

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JEEDS on Blackhawks shots.")
    parser.add_argument(
        "player_ids",
        type=int,
        nargs="+",
        help="One or more player identifiers to estimate.",
    )
    parser.add_argument(
        "game_ids",
        type=int,
        nargs="+",
        help="One or more game identifiers to include in the estimation run.",
    )
    # TODO: last exited recursive code check here

    parser.add_argument(
        "--candidate-skills",
        type=float,
        nargs="+",
        default=None,
        help=f"Optional execution-skill grid for JEEDS in radians (defaults to {DEFAULT_NUM_EXECUTION_SKILLS} values between 0.004 and π/4).",
    )
    parser.add_argument(
        "--num-planning-skills",
        type=int,
        default=25,
        help="Number of planning-skill hypotheses passed to JEEDS.",
    )
    parser.add_argument(
        "--results-folder",
        type=Path,
        default=Path("Experiments/blackhawks-jeeds"),
        help="Directory where JEEDS timing hooks can write logs.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=0,
        help="Seed for the numpy random generator used during estimation.",
    )
    parser.add_argument(
        "--capture-logs",
        action="store_true",
        help="If set, capture intermediate MAP estimates after each shot.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    
    # Support both single and multiple players
    player_ids = args.player_ids if isinstance(args.player_ids, list) else [args.player_ids]
    
    results = estimate_multiple_players(
        player_ids=player_ids,
        game_ids=args.game_ids,
        candidate_skills=args.candidate_skills,
        num_planning_skills=args.num_planning_skills,
        results_folder=args.results_folder,
        rng_seed=args.rng_seed,
        capture_skill_logs=args.capture_logs,
    )
    
    # Print results summary
    print("\n" + "=" * 80)
    print("JEEDS Multi-Player Skill Estimation Results")
    print("=" * 80)
    
    for result in results:
        player_id = result.get("player_id")
        status = result.get("status")
        
        if status == "success":
            print(f"\nPlayer {player_id}:")
            print(f"  Execution Skill: {result['execution_skill']:.4f} rad (lower is better)")
            print(f"  Rationality:     {result['rationality']:.2f} (higher is better, EXPERIMENTAL)")
            print(f"  Shots Used:      {result['num_shots']}")
            
            if "skill_log" in result and result["skill_log"]:
                print(f"  Tracked {len(result['skill_log'])} intermediate estimates")
        else:
            print(f"\nPlayer {player_id}: ERROR")
            print(f"  {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print(f"Completed: {sum(1 for r in results if r['status'] == 'success')}/{len(results)} players")
    print("=" * 80)


if __name__ == "__main__":
    main()
