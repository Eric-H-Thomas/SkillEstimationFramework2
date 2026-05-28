"""Diagnose rationality scaling/capping for legacy vs new xG JEEDS runs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _read_player_ids(players_file: Path) -> list[int]:
    player_ids: list[int] = []
    for raw in players_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.isdigit():
            raise ValueError(f"Invalid player id line '{line}' in {players_file}")
        player_ids.append(int(line))
    if not player_ids:
        raise ValueError(f"No player IDs found in {players_file}")
    return player_ids


def _last_finite(series: pd.Series) -> float | None:
    if series is None:
        return None
    values = series.to_numpy(dtype=float)
    for val in values[::-1]:
        if np.isfinite(val):
            return float(val)
    return None


def _max_finite(series: pd.Series) -> float | None:
    if series is None:
        return None
    values = series.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _pct_ge(values: np.ndarray, threshold: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values >= threshold))


def _season_label(season: int) -> str:
    s = str(season)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:]}"
    return s


def _collect_rows(
    *,
    players: list[int],
    seasons: list[int],
    data_root: Path,
    shot_group: str,
    model: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pid in players:
        for season in seasons:
            csv_path = (
                data_root
                / "players"
                / f"player_{pid}__{model}"
                / "logs"
                / shot_group
                / f"intermediate_estimates_{season}.csv"
            )
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            last_eps = _last_finite(df.get("expected_rationality"))
            last_log10_eps = _last_finite(df.get("log10_expected_rationality"))
            max_log10_eps = _max_finite(df.get("log10_expected_rationality"))
            last_map = _last_finite(df.get("map_rationality"))
            last_log10_map = _last_finite(df.get("log10_map_rationality"))
            shots = _last_finite(df.get("shot_count"))

            rows.append(
                {
                    "player_id": int(pid),
                    "season": int(season),
                    "model": model,
                    "shots": int(shots) if shots is not None else None,
                    "last_eps": last_eps,
                    "last_log10_eps": last_log10_eps,
                    "max_log10_eps": max_log10_eps,
                    "last_map_rationality": last_map,
                    "last_log10_map_rationality": last_log10_map,
                    "csv_path": str(csv_path),
                }
            )
    return rows


def _spread_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
            "iqr": float("nan"),
            "p90_p10": float("nan"),
        }
    q = lambda p: float(np.percentile(values, p))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": q(10),
        "p25": q(25),
        "median": q(50),
        "p75": q(75),
        "p90": q(90),
        "max": float(np.max(values)),
        "iqr": q(75) - q(25),
        "p90_p10": q(90) - q(10),
    }


def _summarize(df: pd.DataFrame, threshold: float) -> dict[str, object]:
    values = df["last_log10_eps"].dropna().to_numpy(dtype=float)
    max_values = df["max_log10_eps"].dropna().to_numpy(dtype=float)

    summary = {
        "files": int(len(df)),
        "last_log10_eps": _spread_stats(values),
        "max_log10_eps": _spread_stats(max_values),
        "pct_last_near_cap": _pct_ge(values, threshold),
        "pct_any_near_cap": _pct_ge(max_values, threshold),
    }
    return summary


def run_diagnostics(
    *,
    players_file: Path,
    seasons: list[int],
    data_root: Path,
    shot_group: str,
    cap_log10: float,
    cap_tolerance: float,
    output_dir: Path,
) -> dict[str, object]:
    players = _read_player_ids(players_file)
    rows: list[dict[str, object]] = []

    for model in ("legacy", "new"):
        rows.extend(
            _collect_rows(
                players=players,
                seasons=seasons,
                data_root=data_root,
                shot_group=shot_group,
                model=model,
            )
        )

    df = pd.DataFrame(rows)
    threshold = float(cap_log10 - cap_tolerance)

    payload: dict[str, object] = {
        "run_metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_root": str(data_root),
            "players_file": str(players_file),
            "shot_group": shot_group,
            "seasons": [int(s) for s in seasons],
            "cap_log10": float(cap_log10),
            "cap_tolerance": float(cap_tolerance),
            "near_cap_threshold": threshold,
        },
        "overall": {},
        "per_season": {},
    }

    for model in ("legacy", "new"):
        model_df = df[df["model"] == model]
        payload["overall"][model] = _summarize(model_df, threshold)

        per_season: dict[str, object] = {}
        for season in seasons:
            s_df = model_df[model_df["season"] == season]
            per_season[str(season)] = _summarize(s_df, threshold)
        payload["per_season"][model] = per_season

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"rationality_scale_diagnostics_{ts}.json"
    csv_path = output_dir / f"rationality_scale_diagnostics_{ts}.csv"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return {
        "rows": int(len(df)),
        "csv": str(csv_path),
        "json": str(json_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose rationality scaling for legacy vs new xG runs.")
    parser.add_argument(
        "--players-file",
        type=Path,
        default=Path("Data/Hockey/forwards23-25.txt"),
        help="Player list (one ID per line).",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[20222023, 20232024],
        help="Seasons to evaluate (default: 20222023 20232024).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data/Hockey/xg_legacy_new_comparison"),
        help="Root directory containing player_*__legacy/new logs.",
    )
    parser.add_argument(
        "--shot-group",
        default="wristshot_snapshot",
        help="Shot group subdirectory under logs/.",
    )
    parser.add_argument(
        "--cap-log10",
        type=float,
        default=4.0,
        help="Log10 cap assumed for rationality grid (default: 4.0).",
    )
    parser.add_argument(
        "--cap-tolerance",
        type=float,
        default=0.05,
        help="Tolerance below cap for flagging near-cap values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/Hockey/xg_legacy_new_comparison/reports"),
        help="Directory for diagnostic CSV/JSON outputs.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_diagnostics(
        players_file=args.players_file,
        seasons=[int(s) for s in args.seasons],
        data_root=args.data_root,
        shot_group=args.shot_group,
        cap_log10=float(args.cap_log10),
        cap_tolerance=float(args.cap_tolerance),
        output_dir=args.output_dir,
    )

    print("Rationality scale diagnostics complete")
    print(f"Rows: {result['rows']}")
    print(f"CSV:  {result['csv']}")
    print(f"JSON: {result['json']}")


if __name__ == "__main__":
    main()
