#!/usr/bin/env python3
# Paper correspondence: Supplement `app:baseball_hyperpriors`; non-paper data-provenance utility.
"""Build the processed Statcast artifact consumed by the MLB experiments.

This is the historical ``BaseballSpaces.getAllData`` shape retained for
non-paper exploration. A fresh fit cannot reproduce the training-time scaler or
batter-index mapping paired with ``final_OP``; the paper requires the canonical
hash-pinned pickle documented in ``Data/Baseball/StatcastData/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Environments.Baseball.dataTake2 import manageData


DEFAULT_DATA_DIR = REPO_ROOT / "Data" / "Baseball" / "StatcastData"
RAW_FILENAMES = ("raw22.csv", "raw21.csv", "raw18_19_20.csv")
OUTPUT_FILENAME = "ProcessedData-From-GivenFiles.pkl"
PROVENANCE_FILENAME = "ProcessedData-From-GivenFiles.provenance.json"
MIN_PLATE_X = -2.13
MAX_PLATE_X = 2.13
MIN_PLATE_Z = -2.50
MAX_PLATE_Z = 6.60
DEFAULT_DELTA = 0.0417


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing processed pickle instead of stopping.",
    )
    parser.add_argument(
        "--allow-unvalidated-refit",
        action="store_true",
        help=(
            "Acknowledge that fitting a fresh scaler/batter mapping is NON-PAPER ONLY "
            "and will not pass the runtime's canonical artifact validation."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not args.allow_unvalidated_refit:
        raise SystemExit(
            "Refusing to refit final_OP preprocessing for a paper artifact. Obtain the canonical "
            "hash-pinned pickle from README.md. For explicitly non-paper exploration only, pass "
            "--allow-unvalidated-refit."
        )
    if args.delta <= 0:
        raise ValueError("--delta must be positive")

    data_dir = args.data_dir.resolve()
    raw_paths = tuple(data_dir / name for name in RAW_FILENAMES)
    missing = [path for path in raw_paths if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing raw Statcast file(s): {names}. Run "
            "Environments/Baseball/downloadStatcastCSVs.py first."
        )

    output_path = data_dir / OUTPUT_FILENAME
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it."
        )

    all_data, batter_indices, standardizer = manageData([str(path) for path in raw_paths])

    targets_plate_x_feet = np.arange(MIN_PLATE_X, MAX_PLATE_X, args.delta)
    targets_plate_z_feet = np.arange(MIN_PLATE_Z, MAX_PLATE_Z, args.delta)
    # plate_x and plate_z are columns 3 and 4 in dataTake2.manageData's scaler.
    model_targets_plate_x = (
        targets_plate_x_feet - standardizer.mean_[3]
    ) / standardizer.scale_[3]
    model_targets_plate_z = (
        targets_plate_z_feet - standardizer.mean_[4]
    ) / standardizer.scale_[4]

    # Preserve the original nested-loop order: x is the outer index, z inner.
    possible_targets_feet = np.column_stack(
        (
            np.repeat(targets_plate_x_feet, len(targets_plate_z_feet)),
            np.tile(targets_plate_z_feet, len(targets_plate_x_feet)),
        )
    )
    possible_targets_for_model = np.column_stack(
        (
            np.repeat(model_targets_plate_x, len(model_targets_plate_z)),
            np.tile(model_targets_plate_z, len(model_targets_plate_x)),
        )
    )

    payload = [
        all_data,
        batter_indices,
        model_targets_plate_x,
        model_targets_plate_z,
        possible_targets_feet,
        possible_targets_for_model,
    ]
    data_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        # The outer list is required by HJEEDS.baseball_pitch.load_processed_statcast.
        pickle.dump([payload], handle)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "delta_feet": args.delta,
        "model_compatibility": {
            "validated_against_training_reference": False,
            "status": "non-paper-refit",
        },
        "processed_rows": int(len(all_data)),
        "target_grid_shape": [
            int(len(targets_plate_x_feet)),
            int(len(targets_plate_z_feet)),
        ],
        "raw_files": [
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in raw_paths
        ],
        "output": {
            "filename": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        },
    }
    provenance_path = data_dir / PROVENANCE_FILENAME
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Wrote {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
