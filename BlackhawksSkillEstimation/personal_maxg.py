"""Personal shot-pool MAXG for one player's own shots.

For each season, blur that player's post-shot xG surfaces across a grid of
execution sigmas and sum two readouts:

- MAXG is the maximum of the raw blurred surface, the value of aiming at the
  best location at that sigma.
- Policy expectation is the softmax expectation of that same raw surface.
  The softmax uses peak-above-average normalization, which is how the saved
  lambdas were estimated. The y-axis stays in goals.

The blur is ``HJEEDS.hockey_adapter.blur_utility``, the Gaussian filter the
JEEDS and HJEEDS estimates were fit under. These totals are not on the scale
of the benchmark ``convolve2d`` MAXG scores.

Personal sigmas come from the pooled HJEEDS agent table. That run stores both
the independent JEEDS mean and the hierarchical mean for each player.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from HJEEDS.hockey_adapter import HockeySituation, blur_utility, prepare_hockey_situations
from HJEEDS.hockey_pipeline import _filter_shot_group


DEFAULT_SEASONS = (20212022, 20222023, 20232024, 20242025, 20252026)
DEFAULT_SHOT_GROUP = "wristshot_snapshot"
DEFAULT_POOLED_RESULTS = Path(
    "Data/Hockey/experiments/hjeeds_pooled_20212022_20252026/agent_level_results.csv"
)
WIDE_SIGMA_MIN = 0.01
WIDE_SIGMA_MAX = 0.20
N_GRID = 10

SWEEP_EMPIRICAL = "empirical_jeeds"
SWEEP_WIDE = "wide"
SWEEP_PERSONAL = "personal"
MARKER_GRID = "grid"
MARKER_POOLED = "pooled_hierarchical"
MARKER_JEEDS = "pooled_jeeds"
SERIES_MAXG = "maxg"
SERIES_POLICY = "policy_expectation"

_COLOR_MAXG = "#4C78A8"
_COLOR_POLICY = "#F58518"
_COLOR_POOLED = "#E45756"
_COLOR_JEEDS = "#54A24B"


@dataclass(frozen=True)
class ShotRecord:
    """One surviving shot and the goal flag aligned with it."""

    situation: HockeySituation
    is_goal: bool


@dataclass(frozen=True)
class ScoreRequest:
    """One (sweep, marker) evaluation of a blurred season pool."""

    sweep: str
    marker: str
    policy_lambda: float


def maxg_value(raw: np.ndarray) -> float:
    """Maximum of a raw blurred xG surface."""

    return float(np.max(_finite_values(raw)))


def policy_expectation(raw: np.ndarray, lambda_value: float) -> float:
    """Expected raw xG under a peak-above-average softmax.

    Normalization is applied only to the softmax scores. The returned value
    is the expectation of the un-normalized surface, in xG. Saved lambdas were
    estimated with this normalization on, so the readout does not consult
    ``BH_EV_NORMALIZE``.
    """

    values = _finite_values(raw)
    scale = float(np.max(values) - np.mean(values))
    scores = values / scale if scale > 1e-12 else values
    scores = float(lambda_value) * scores
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("softmax weights are not usable")
    return float(np.sum((weights / total) * values))


def score_surfaces(surfaces: Sequence[np.ndarray], lambda_value: float) -> tuple[float, float]:
    """Sum MAXG and policy expectation over blurred shots."""

    maxg = 0.0
    policy = 0.0
    for surface in surfaces:
        maxg += maxg_value(surface)
        policy += policy_expectation(surface, lambda_value)
    return maxg, policy


def empirical_sigma_grid(results: pd.DataFrame, n_points: int = N_GRID) -> np.ndarray:
    """Inclusive linear grid from the smallest to largest pooled JEEDS mean sigma."""

    if "jeeds_mean_sigma" not in results.columns:
        raise ValueError("pooled results are missing jeeds_mean_sigma")
    sigmas = pd.to_numeric(results["jeeds_mean_sigma"], errors="coerce")
    if "jeeds_status" in results.columns:
        sigmas = sigmas[results["jeeds_status"].astype(str) == "ok"]
    sigmas = sigmas[np.isfinite(sigmas)]
    if sigmas.empty:
        raise RuntimeError("no usable pooled JEEDS sigmas")
    if int(n_points) < 2:
        raise ValueError("n_points must be at least 2")
    return np.linspace(float(sigmas.min()), float(sigmas.max()), int(n_points))


def wide_sigma_grid(n_points: int = N_GRID) -> np.ndarray:
    """Inclusive linear sensitivity grid from 0.01 to 0.20 radians."""

    if int(n_points) < 2:
        raise ValueError("n_points must be at least 2")
    return np.linspace(WIDE_SIGMA_MIN, WIDE_SIGMA_MAX, int(n_points))


def realized_shot_xg(
    utility_grid: np.ndarray,
    target_angles: np.ndarray,
    executed_action: np.ndarray,
) -> float:
    """Unblurred angular utility at the cell nearest the observed angle."""

    utilities = np.asarray(utility_grid, dtype=float).reshape(-1)
    targets = np.asarray(target_angles, dtype=float)
    executed = np.asarray(executed_action, dtype=float).reshape(-1)
    if targets.shape != (utilities.size, 2):
        raise ValueError(
            f"target angles shape {targets.shape} does not match utility size {utilities.size}"
        )
    if executed.shape != (2,):
        raise ValueError(f"executed action must have shape (2,), got {executed.shape}")
    nearest = int(np.argmin(np.sum((targets - executed) ** 2, axis=1)))
    value = float(utilities[nearest])
    if not np.isfinite(value):
        raise ValueError("realized post-shot xG is not finite")
    return value


def player_pooled_profile(results: pd.DataFrame, player_id: int) -> tuple[float, float]:
    """Pooled hierarchical mean sigma and the matching mean lambda."""

    required = {"player_id", "hierarchical_mean_sigma", "hierarchical_mean_log_lambda"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"pooled results are missing columns: {sorted(missing)}")
    rows = results[pd.to_numeric(results["player_id"], errors="coerce") == int(player_id)]
    if "hierarchical_status" in results.columns:
        rows = rows[rows["hierarchical_status"].astype(str) == "ok"]
    if rows.empty:
        raise RuntimeError(f"no pooled hierarchical estimate for player {player_id}")
    row = rows.iloc[0]
    sigma = float(row["hierarchical_mean_sigma"])
    log_lambda = float(row["hierarchical_mean_log_lambda"])
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(log_lambda):
        raise RuntimeError(f"pooled hierarchical estimate for player {player_id} is not usable")
    return sigma, float(np.exp(log_lambda))


def player_jeeds_profile(results: pd.DataFrame, player_id: int) -> tuple[float, float]:
    """Pooled JEEDS mean sigma and the matching mean lambda from the same agent table."""

    required = {"player_id", "jeeds_mean_sigma", "jeeds_mean_log_lambda"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"pooled results are missing columns: {sorted(missing)}")
    rows = results[pd.to_numeric(results["player_id"], errors="coerce") == int(player_id)]
    if "jeeds_status" in results.columns:
        rows = rows[rows["jeeds_status"].astype(str) == "ok"]
    if rows.empty:
        raise RuntimeError(f"no pooled JEEDS estimate for player {player_id}")
    row = rows.iloc[0]
    sigma = float(row["jeeds_mean_sigma"])
    log_lambda = float(row["jeeds_mean_log_lambda"])
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(log_lambda):
        raise RuntimeError(f"pooled JEEDS estimate for player {player_id} is not usable")
    return sigma, float(np.exp(log_lambda))


def format_season(season: int) -> str:
    text = str(int(season))
    if len(text) != 8:
        return text
    return f"{text[:4]}-{text[6:]}"


def build_shot_pool(shots_df: pd.DataFrame, shot_maps: dict[int, dict[str, object]]) -> list[ShotRecord]:
    """Filter to situations the hockey likelihood would score, and keep goal flags."""

    if "shot_is_goal" not in shots_df.columns and not shots_df.empty:
        raise ValueError("shots_df is missing shot_is_goal")
    situations = prepare_hockey_situations(shots_df, shot_maps)
    goals: dict[int, bool] = {}
    for _, row in shots_df.iterrows():
        event_id = int(row["event_id"])
        goals.setdefault(event_id, _as_goal(row["shot_is_goal"]))
    records: list[ShotRecord] = []
    for situation in situations:
        if situation.event_id not in goals:
            raise RuntimeError(f"shot {situation.event_id} survived the angular filter without a goal flag")
        records.append(ShotRecord(situation=situation, is_goal=goals[situation.event_id]))
    return records


def pool_totals(records: Sequence[ShotRecord]) -> tuple[int, float, float]:
    """Shot count, goals, and realized post-shot xG on the surviving pool."""

    realized = 0.0
    goals = 0.0
    for record in records:
        situation = record.situation
        realized += realized_shot_xg(
            situation.utility_grid,
            situation.target_angles,
            situation.executed_action,
        )
        goals += float(record.is_goal)
    return len(records), goals, realized


def blur_pool(records: Sequence[ShotRecord], sigma: float) -> list[np.ndarray]:
    """Raw-xG Gaussian blur of every shot at one execution sigma."""

    return [
        blur_utility(record.situation.utility_grid, float(sigma), record.situation.bin_size)
        for record in records
    ]


def _players_root(data_dir: Path) -> Path:
    return data_dir if data_dir.name == "players" else data_dir / "players"


def season_data_paths(data_dir: Path, player_id: int, season: int) -> tuple[Path, Path]:
    """Cached shots parquet and shot-map npz for one player-season."""

    data = _players_root(data_dir) / f"player_{player_id}" / "data"
    tag = int(season)
    return data / f"shots_{tag}.parquet", data / f"shot_maps_{tag}.npz"


def results_shot_group(results_path: Path) -> str | None:
    """Shot group recorded beside an HJEEDS agent table, if the manifest has one."""

    manifest_path = results_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("shot_group")
    if recorded is None or str(recorded) == "":
        return None
    return str(recorded)


def score_player(
    *,
    player_id: int,
    seasons: Sequence[int],
    shot_group: str,
    pooled_results: pd.DataFrame,
    data_dir: Path,
    n_points: int = N_GRID,
) -> pd.DataFrame:
    """Score one player's season shot pools on both sigma sweeps and both personal sigmas.

    The JEEDS and hierarchical markers are that player's rows in ``pooled_results``.
    Seasons with no cached shots or no surviving shots are skipped.
    """

    empirical = empirical_sigma_grid(pooled_results, n_points)
    wide = wide_sigma_grid(n_points)
    pooled_sigma, pooled_lambda = player_pooled_profile(pooled_results, player_id)
    jeeds_sigma, jeeds_lambda = player_jeeds_profile(pooled_results, player_id)
    print(
        f"player {player_id}: pooled JEEDS σ={jeeds_sigma:.5f} λ={jeeds_lambda:.3f}; "
        f"hierarchical σ={pooled_sigma:.5f} λ={pooled_lambda:.3f}",
        flush=True,
    )
    rows: list[dict[str, object]] = []

    for season in seasons:
        started = time.perf_counter()
        season_label = format_season(int(season))
        shots_path, maps_path = season_data_paths(data_dir, player_id, int(season))
        if not shots_path.exists() or not maps_path.exists():
            print(f"season {season_label}: no shot files, skipping", flush=True)
            continue
        print(f"season {season}: loading shots", flush=True)
        shots, shot_maps = load_player_data(player_id, [int(season)], data_dir=data_dir)
        shots = _filter_shot_group(shots, shot_group)
        records = build_shot_pool(shots, shot_maps)
        if not records:
            print(f"season {season_label}: no usable {shot_group} shots, skipping", flush=True)
            continue
        n_shots, goals, realized = pool_totals(records)
        requests: dict[float, list[ScoreRequest]] = defaultdict(list)
        for sigma in empirical:
            requests[_sigma_key(sigma)].append(ScoreRequest(SWEEP_EMPIRICAL, MARKER_GRID, pooled_lambda))
        for sigma in wide:
            requests[_sigma_key(sigma)].append(ScoreRequest(SWEEP_WIDE, MARKER_GRID, pooled_lambda))
        requests[_sigma_key(pooled_sigma)].append(ScoreRequest(SWEEP_PERSONAL, MARKER_POOLED, pooled_lambda))
        requests[_sigma_key(jeeds_sigma)].append(ScoreRequest(SWEEP_PERSONAL, MARKER_JEEDS, jeeds_lambda))

        print(
            f"season {format_season(int(season))}: scoring {n_shots} shots at {len(requests)} sigmas",
            flush=True,
        )
        for sigma, pending in requests.items():
            surfaces = blur_pool(records, sigma)
            maxg = sum(maxg_value(surface) for surface in surfaces)
            for request in pending:
                policy = sum(policy_expectation(surface, request.policy_lambda) for surface in surfaces)
                rows.extend(
                    _score_rows(
                        player_id=player_id,
                        season=int(season),
                        request=request,
                        sigma=sigma,
                        maxg=maxg,
                        policy=policy,
                        n_shots=n_shots,
                        goals=goals,
                        realized_xg=realized,
                    )
                )
        elapsed = time.perf_counter() - started
        print(
            f"season {format_season(int(season))}: n={n_shots} goals={goals:.0f} "
            f"realized_xg={realized:.2f} ({elapsed:.1f}s)",
            flush=True,
        )

    if not rows:
        raise RuntimeError(f"player {player_id} has no seasons with usable {shot_group} shots")
    frame = pd.DataFrame(rows)
    sweep_order = {SWEEP_EMPIRICAL: 0, SWEEP_WIDE: 1, SWEEP_PERSONAL: 2}
    series_order = {SERIES_MAXG: 0, SERIES_POLICY: 1}
    frame["_sweep_order"] = frame["sweep"].map(sweep_order)
    frame["_series_order"] = frame["series"].map(series_order)
    frame = frame.sort_values(
        ["season", "_sweep_order", "sigma", "_series_order", "marker"],
        kind="mergesort",
    ).drop(columns=["_sweep_order", "_series_order"])
    return frame.reset_index(drop=True)


def plot_sweep(
    scores: pd.DataFrame,
    sweep: str,
    output_path: Path,
    *,
    player_id: int,
    fixed_lambda: float,
) -> None:
    """One figure, one panel per season, for a single sigma sweep."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep_rows = scores[scores["sweep"] == sweep]
    personal = scores[scores["sweep"] == SWEEP_PERSONAL]
    if sweep_rows.empty:
        raise RuntimeError(f"no rows for sweep {sweep}")
    seasons = list(dict.fromkeys(int(season) for season in sweep_rows["season"]))
    fig, axes = plt.subplots(
        1,
        len(seasons),
        figsize=(3.8 * len(seasons), 5.2),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    axes_flat = list(axes[0])
    ylim = _shared_ylim(sweep_rows, personal)
    xlim = _shared_xlim(sweep_rows, personal)

    for index, (axis, season) in enumerate(zip(axes_flat, seasons)):
        season_grid = sweep_rows[sweep_rows["season"] == season]
        season_personal = personal[personal["season"] == season]
        labeled = index == 0
        _scatter_series(
            axis,
            season_grid,
            SERIES_MAXG,
            color=_COLOR_MAXG,
            marker="o",
            size=36,
            label="MAXG" if labeled else None,
        )
        _scatter_series(
            axis,
            season_grid,
            SERIES_POLICY,
            color=_COLOR_POLICY,
            marker="o",
            size=36,
            label="policy expectation" if labeled else None,
        )
        _personal_markers(axis, season_personal, labeled=labeled)
        n_shots = int(season_grid["n_shots"].iloc[0])
        goals = float(season_grid["goals"].iloc[0])
        realized = float(season_grid["realized_xg"].iloc[0])
        axis.axhline(
            goals,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="goals" if labeled else None,
        )
        axis.axhline(
            realized,
            color="#666666",
            linestyle=":",
            linewidth=1.2,
            label="realized post-shot xG" if labeled else None,
        )
        axis.set_title(f"{format_season(season)}   n={n_shots}")
        axis.grid(True, alpha=0.25)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.tick_params(axis="y", labelleft=True, labelsize=8)

    axes_flat[0].set_ylabel("total expected goals")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
    )
    title = _sweep_title(sweep, sweep_rows)
    jeeds_lambda = _marker_lambda(personal, MARKER_JEEDS)
    lambda_note = f"orange curve uses hierarchical λ = {fixed_lambda:.2f}"
    if jeeds_lambda is not None:
        lambda_note += f"; green rings use JEEDS λ = {jeeds_lambda:.2f}"
    fig.suptitle(f"Player {player_id}, {title}\n{lambda_note}")
    fig.text(
        0.5,
        0.145,
        "execution sigma (radians); smaller is tighter",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.24, top=0.84, wspace=0.42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def write_outputs(
    scores: pd.DataFrame,
    output_dir: Path,
    *,
    player_id: int,
    fixed_lambda: float,
) -> tuple[Path, Path, Path]:
    """Write the score table and the two sweep figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scores.csv"
    empirical_path = output_dir / "empirical_jeeds_sweep.png"
    wide_path = output_dir / "wide_sensitivity_sweep.png"
    scores.to_csv(csv_path, index=False)
    plot_sweep(scores, SWEEP_EMPIRICAL, empirical_path, player_id=player_id, fixed_lambda=fixed_lambda)
    plot_sweep(scores, SWEEP_WIDE, wide_path, player_id=player_id, fixed_lambda=fixed_lambda)
    return csv_path, empirical_path, wide_path


def run(
    *,
    player_id: int,
    seasons: Sequence[int],
    shot_group: str,
    pooled_results_path: Path,
    data_dir: Path,
    output_dir: Path,
    n_points: int = N_GRID,
) -> pd.DataFrame:
    """Score one player and write the CSV and both figures."""

    if not pooled_results_path.exists():
        raise FileNotFoundError(f"missing pooled results: {pooled_results_path}")
    recorded_group = results_shot_group(pooled_results_path)
    if recorded_group is not None and recorded_group != shot_group:
        raise ValueError(
            f"{pooled_results_path} is a {recorded_group} run; requested shot group is {shot_group}"
        )
    pooled_results = pd.read_csv(pooled_results_path)
    _, fixed_lambda = player_pooled_profile(pooled_results, player_id)
    scores = score_player(
        player_id=player_id,
        seasons=seasons,
        shot_group=shot_group,
        pooled_results=pooled_results,
        data_dir=data_dir,
        n_points=n_points,
    )
    csv_path, empirical_path, wide_path = write_outputs(
        scores,
        output_dir,
        player_id=player_id,
        fixed_lambda=fixed_lambda,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {empirical_path}")
    print(f"wrote {wide_path}")
    return scores


def _finite_values(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("surface has no finite cells")
    return values


def _as_goal(value: object) -> bool:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        return bool(float(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes"}:
        return True
    if text in {"0", "false", "f", "no", ""}:
        return False
    raise ValueError(f"unrecognized shot_is_goal value: {value!r}")


def _sigma_key(sigma: float) -> float:
    return float(sigma)


def _score_rows(
    *,
    player_id: int,
    season: int,
    request: ScoreRequest,
    sigma: float,
    maxg: float,
    policy: float,
    n_shots: int,
    goals: float,
    realized_xg: float,
) -> list[dict[str, object]]:
    shared = {
        "player_id": player_id,
        "season": season,
        "sweep": request.sweep,
        "marker": request.marker,
        "sigma": sigma,
        "n_shots": n_shots,
        "goals": goals,
        "realized_xg": realized_xg,
    }
    return [
        {**shared, "series": SERIES_MAXG, "lambda": np.nan, "value": maxg},
        {**shared, "series": SERIES_POLICY, "lambda": request.policy_lambda, "value": policy},
    ]


def _scatter_series(
    axis,
    frame: pd.DataFrame,
    series: str,
    *,
    color: str,
    marker: str,
    size: float,
    label: str | None,
) -> None:
    rows = frame[frame["series"] == series]
    axis.scatter(
        rows["sigma"],
        rows["value"],
        c=color,
        marker=marker,
        s=size,
        label=label,
        zorder=3,
    )


def _personal_markers(axis, frame: pd.DataFrame, *, labeled: bool) -> None:
    styles = (
        (MARKER_POOLED, _COLOR_POOLED, "MAXG at pooled hierarchical σ", "policy at pooled hierarchical σ"),
        (MARKER_JEEDS, _COLOR_JEEDS, "MAXG at pooled JEEDS σ", "policy at pooled JEEDS σ"),
    )
    for marker, color, maxg_label, policy_label in styles:
        rows = frame[frame["marker"] == marker]
        maxg = rows[rows["series"] == SERIES_MAXG]
        policy = rows[rows["series"] == SERIES_POLICY]
        if not maxg.empty:
            axis.scatter(
                maxg["sigma"],
                maxg["value"],
                c=color,
                marker="*",
                s=180,
                label=maxg_label if labeled else None,
                zorder=5,
            )
        if not policy.empty:
            axis.scatter(
                policy["sigma"],
                policy["value"],
                facecolors="none",
                edgecolors=color,
                marker="o",
                s=90,
                linewidths=1.8,
                label=policy_label if labeled else None,
                zorder=4,
            )


def _marker_lambda(personal: pd.DataFrame, marker: str) -> float | None:
    """Lambda used for one personal policy marker, if that marker was scored."""

    rows = personal[(personal["marker"] == marker) & (personal["series"] == SERIES_POLICY)]
    if rows.empty:
        return None
    value = float(rows["lambda"].iloc[0])
    if not np.isfinite(value):
        return None
    return value


def _shared_ylim(sweep_rows: pd.DataFrame, personal: pd.DataFrame) -> tuple[float, float]:
    pieces = [
        sweep_rows["value"].to_numpy(dtype=float),
        personal["value"].to_numpy(dtype=float),
        sweep_rows["goals"].to_numpy(dtype=float),
        sweep_rows["realized_xg"].to_numpy(dtype=float),
    ]
    values = np.concatenate([piece for piece in pieces if piece.size])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("no finite scores to set the y-axis")
    low = float(np.min(values))
    high = float(np.max(values))
    pad = 0.05 * (high - low) if high > low else 1.0
    lower = low - pad
    if low >= 0.0:
        lower = max(0.0, lower)
    return lower, high + pad


def _shared_xlim(sweep_rows: pd.DataFrame, personal: pd.DataFrame) -> tuple[float, float]:
    sigmas = np.concatenate(
        [
            sweep_rows["sigma"].to_numpy(dtype=float),
            personal["sigma"].to_numpy(dtype=float),
        ]
    )
    sigmas = sigmas[np.isfinite(sigmas)]
    low = float(np.min(sigmas))
    high = float(np.max(sigmas))
    pad = 0.06 * (high - low) if high > low else 0.01
    return low - pad, high + pad


def _sweep_title(sweep: str, sweep_rows: pd.DataFrame) -> str:
    sigmas = np.sort(sweep_rows["sigma"].unique())
    span = f"σ from {float(sigmas[0]):.4f} to {float(sigmas[-1]):.4f}"
    if sweep == SWEEP_EMPIRICAL:
        return f"empirical JEEDS sweep ({span})"
    if sweep == SWEEP_WIDE:
        return f"wide sensitivity sweep ({span})"
    return sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-id", type=int, default=950160)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--shot-group", default=DEFAULT_SHOT_GROUP)
    parser.add_argument("--pooled-results", type=Path, default=DEFAULT_POOLED_RESULTS)
    parser.add_argument("--data-dir", type=Path, default=Path("Data/Hockey"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-points", type=int, default=N_GRID)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("Data/Hockey/experiments/personal_maxg") / f"player_{args.player_id}"
    run(
        player_id=args.player_id,
        seasons=list(dict.fromkeys(args.seasons)),
        shot_group=args.shot_group,
        pooled_results_path=args.pooled_results,
        data_dir=args.data_dir,
        output_dir=output_dir,
        n_points=args.n_points,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
