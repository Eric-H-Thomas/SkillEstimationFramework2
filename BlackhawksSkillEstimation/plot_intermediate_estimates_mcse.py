"""Plot MCSE intermediate estimates (2D execution skill + rationality)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_mcse_intermediate_estimates(csv_path: Path | str) -> dict[str, list]:
    csv_path = Path(csv_path)
    columns = [
        "shot_count",
        "ees_y",
        "ees_z",
        "map_execution_skill_y",
        "map_execution_skill_z",
        "rho_ees",
        "map_rho",
        "expected_rationality",
        "map_rationality",
        "log10_expected_rationality",
        "log10_map_rationality",
    ]
    data: dict[str, list] = {col: [] for col in columns}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for col in columns:
                val = row.get(col, "")
                if col == "shot_count":
                    data[col].append(int(float(val)) if val not in ("", None) else 0)
                else:
                    try:
                        data[col].append(float(val) if val not in ("", None) else float("nan"))
                    except ValueError:
                        data[col].append(float("nan"))
    return data


def plot_intermediate_estimates_mcse(
    csv_path: Path | str,
    *,
    show: bool = True,
    include_map_estimates: bool = True,
) -> Path | None:
    """Stacked convergence plot: execution skill y/z (top/bottom), rationality below."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    data = load_mcse_intermediate_estimates(csv_path)
    shots = np.asarray(data["shot_count"], dtype=float)
    if len(shots) == 0:
        return None

    ees_y = np.asarray(data["ees_y"], dtype=float)
    ees_z = np.asarray(data["ees_z"], dtype=float)
    map_y = np.asarray(data["map_execution_skill_y"], dtype=float)
    map_z = np.asarray(data["map_execution_skill_z"], dtype=float)
    log10_eps = np.asarray(data["log10_expected_rationality"], dtype=float)
    log10_map = np.asarray(data["log10_map_rationality"], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(shots, ees_y, label="EES", color="C0")
    if include_map_estimates:
        axes[0].plot(shots, map_y, label="MAP", color="C1", linestyle="--")
    axes[0].set_ylabel("Execution skill Y (rad)")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"MCSE convergence — {csv_path.name}")
    axes[0].invert_yaxis()

    axes[1].plot(shots, ees_z, label="EES", color="C0")
    if include_map_estimates:
        axes[1].plot(shots, map_z, label="MAP", color="C1", linestyle="--")
    axes[1].set_ylabel("Execution skill Z (rad)")
    axes[1].legend(loc="upper right")
    axes[1].invert_yaxis()

    axes[2].plot(shots, log10_eps, label="log10 EPS", color="C2")
    if include_map_estimates:
        axes[2].plot(shots, log10_map, label="log10 MAP λ", color="C3", linestyle="--")
    axes[2].set_ylabel("log10 rationality (λ)")
    axes[2].set_xlabel("Shots processed")
    axes[2].legend(loc="upper right")

    for ax in axes[:2]:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    fig.tight_layout()
    out_path = csv_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path
