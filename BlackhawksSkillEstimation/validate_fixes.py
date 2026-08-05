"""Before/after check for the MCSE lambda fix and the JEEDS EV-blur fix.

Runs both estimators on a handful of real player-seasons under the old and new
behaviour and reports whether the estimates now respond to data. Writes into
``player_<id>__validation`` directories so the synced cluster results are left
untouched.

Usage:
    PYTHONPATH=. python BlackhawksSkillEstimation/validate_fixes.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ROOTS = {"legacy": REPO_ROOT / "Data" / "Hockey", "new": REPO_ROOT / "Data" / "Hockey_xg_new"}

# Prior mean of lambda on a log10-uniform [0, 4] grid. An estimate pinned here is
# an estimate that has learned nothing.
PRIOR_MEAN_LOG10_LAMBDA = float(np.log10(np.mean(np.power(10.0, np.linspace(0, 4, 500)))))


def _players_in_root(root: Path, season: int) -> list[int]:
    found = []
    for player_dir in sorted((root / "players").glob("player_*")):
        if "__" in player_dir.name:
            continue
        csv_path = (
            player_dir / "logs" / "mcse" / "wristshot_snapshot" / f"intermediate_estimates_{season}.csv"
        )
        if not csv_path.exists():
            continue
        if len(pd.read_csv(csv_path)) < 140:
            continue
        found.append(int(player_dir.name.split("_")[1]))
    return found


def _pick_players(roots: dict[str, Path], season: int, n: int) -> list[int]:
    """Players with enough data in every selected root, so runs are comparable."""
    eligible: set[int] | None = None
    for root in roots.values():
        in_root = set(_players_in_root(root, season))
        eligible = in_root if eligible is None else (eligible & in_root)
    if not eligible:
        raise SystemExit(
            f"No players with >=140 shots in season {season} across roots: "
            f"{', '.join(r.name for r in roots.values())}"
        )
    return sorted(eligible)[:n]


def run_mcse(
    player_ids: list[int], season: int, root: Path, mode: str, particles: int, max_shots: int
) -> pd.DataFrame:
    from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
    from BlackhawksSkillEstimation.BlackhawksMCSE import estimate_player_skill

    rows = []
    for player_id in player_ids:
        df, shot_maps = load_player_data(player_id, [season], data_dir=str(root))
        if max_shots:
            df = df.head(max_shots)
        print(f"  [mcse/{root.name}/{mode}] player {player_id}: {len(df)} shots...", flush=True)
        result = estimate_player_skill(
            player_id=player_id,
            seasons=[season],
            per_season=True,
            num_particles=particles,
            lambda_mode=mode,
            save_intermediate_csv=False,
            confirm=False,
            offline_data=(df, shot_maps),
            shot_group="wristshot_snapshot",
            data_dir=str(root),
            player_dir_name=f"player_{player_id}__validation",
        )
        season_result = (result.get("per_season_results") or {}).get(season, result)
        eps = season_result.get("eps")
        rows.append(
            {
                "player_id": player_id,
                "lambda_mode": mode,
                "ees_y": season_result.get("ees_y"),
                "ees_z": season_result.get("ees_z"),
                "rho_ees": season_result.get("rho_ees"),
                "log10_eps": float(np.log10(eps)) if eps and eps > 0 else np.nan,
                "num_shots": season_result.get("num_shots"),
            }
        )
    return pd.DataFrame(rows)


def run_jeeds(
    player_ids: list[int],
    season: int,
    root: Path,
    cap: str,
    max_shots: int,
    normalize: str = "0",
) -> pd.DataFrame:
    os.environ["BH_EV_BLUR_MAX_SIGMA_BINS"] = cap
    os.environ["BH_EV_NORMALIZE"] = normalize
    # The cap is read at import time, so reload the module to pick up the new value.
    import importlib

    import BlackhawksSkillEstimation.BlackhawksJEEDS as jeeds_module

    importlib.reload(jeeds_module)

    rows = []
    for player_id in player_ids:
        df, shot_maps = jeeds_module.load_player_data(player_id, [season], data_dir=str(root))
        if max_shots:
            df = df.head(max_shots)
        print(f"  [jeeds/{root.name}/cap={cap}] player {player_id}: {len(df)} shots...", flush=True)
        result = jeeds_module.estimate_player_skill(
            player_id=player_id,
            seasons=[season],
            per_season=True,
            confirm=False,
            offline_data=(df, shot_maps),
            shot_group="wristshot_snapshot",
            data_dir=str(root),
            player_dir_name=f"player_{player_id}__validation",
        )
        season_result = (result.get("per_season_results") or {}).get(season, result)
        rows.append(
            {
                "player_id": player_id,
                "blur_cap_bins": cap,
                "ev_normalized": normalize,
                "ees": season_result.get("ees"),
                "log10_eps": season_result.get("log10_eps"),
                "num_shots": season_result.get("num_shots"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-players", type=int, default=3)
    parser.add_argument("--season", type=int, default=20242025)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--max-shots", type=int, default=0, help="0 uses every shot")
    parser.add_argument("--roots", nargs="+", default=list(ROOTS), choices=list(ROOTS))
    parser.add_argument("--skip-jeeds", action="store_true")
    parser.add_argument("--skip-mcse", action="store_true")
    args = parser.parse_args()
    roots = {k: ROOTS[k] for k in args.roots}

    pd.set_option("display.width", 200)

    players = _pick_players(roots, args.season, args.n_players)
    print(
        f"Validating on players {players}, season {args.season}, "
        f"roots {', '.join(roots)}\n"
    )

    if not args.skip_mcse:
        print("=== MCSE: does rationality respond to the data now? ===")
        print(f"(prior mean sits at log10 lambda = {PRIOR_MEAN_LOG10_LAMBDA:.3f}; "
              "the old code pinned every player there)\n")
        mcse_frames = []
        for xg_model, root in roots.items():
            for mode in ["fixed_grid", "estimated"]:
                frame = run_mcse(
                    players, args.season, root, mode, args.particles, args.max_shots
                )
                frame["xg_model"] = xg_model
                mcse_frames.append(frame)
        mcse = pd.concat(mcse_frames, ignore_index=True)
        print()
        print(mcse.to_string(index=False))

        print("\n--- spread of log10 rationality across players (0 = no signal) ---")
        for (xg_model, mode), group in mcse.groupby(["xg_model", "lambda_mode"]):
            spread = group["log10_eps"].max() - group["log10_eps"].min()
            offset = group["log10_eps"].mean() - PRIOR_MEAN_LOG10_LAMBDA
            print(
                f"  {xg_model:7s} {mode:11s} range={spread:.3f}  "
                f"mean offset from prior={offset:+.3f}"
            )

    if args.skip_jeeds:
        return

    print("\n=== JEEDS: blur clamp and EV normalization ===")
    variants = [
        ("1.0", "0", "A: as-shipped (clamp on, raw EV)"),
        ("1e9", "0", "B: clamp lifted, raw EV"),
        ("1e9", "1", "C: clamp lifted + EV normalized"),
    ]
    jeeds_frames = []
    for xg_model, root in roots.items():
        for cap, normalize, _ in variants:
            frame = run_jeeds(players, args.season, root, cap, args.max_shots, normalize)
            frame["xg_model"] = xg_model
            jeeds_frames.append(frame)
    jeeds = pd.concat(jeeds_frames, ignore_index=True)
    print()
    print(jeeds.to_string(index=False))

    print("\n--- per variant: median estimate, and do the two xG models agree? ---")
    for cap, normalize, label in variants:
        sub = jeeds[(jeeds["blur_cap_bins"] == cap) & (jeeds["ev_normalized"] == normalize)]
        parts = []
        for xg_model in roots:
            group = sub[sub["xg_model"] == xg_model]
            if group.empty:
                continue
            parts.append(
                f"{xg_model}: ees={group['ees'].median():.4f} "
                f"log10_lambda={group['log10_eps'].median():.3f} "
                f"(spread {group['log10_eps'].max() - group['log10_eps'].min():.3f})"
            )
        gap = ""
        if len(roots) > 1:
            medians = [
                sub[sub["xg_model"] == m]["ees"].median() for m in roots if not sub[sub["xg_model"] == m].empty
            ]
            if len(medians) == 2:
                gap = f"  | legacy-vs-new ees gap = {abs(medians[0] - medians[1]):.4f}"
        print(f"  {label}\n      " + "   ".join(parts) + gap)


if __name__ == "__main__":
    main()
