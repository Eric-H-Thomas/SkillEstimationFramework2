"""Guard against running on outdated Blackhawks xG value maps.

The current post-shot xG model stores each shot's value map on a 31 x 51 net
grid (rows are Z, columns are Y). The first model iteration stored 72 x 120
maps. Both kinds of file share the same names and keys, so a data tree that
still holds the old maps loads without error and shifts sigma estimates by
about 11%. These checks make that mismatch a hard failure.
"""
from __future__ import annotations

import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

CURRENT_SHOT_MAP_SHAPE: tuple[int, int] = (31, 51)
LEGACY_SHOT_MAP_SHAPE: tuple[int, int] = (72, 120)


class ShotMapFormatError(RuntimeError):
    """Raised when shot maps do not use the expected xG value-map grid."""


def _describe(shape: tuple[int, ...]) -> str:
    if tuple(shape) == LEGACY_SHOT_MAP_SHAPE:
        return f"{shape} (legacy xG maps)"
    return str(shape)


def read_value_map_shape(path: Path | str) -> tuple[int, int] | None:
    """Return the per-shot value-map shape stored in a shot-maps NPZ.

    Only the array header is read, so this is cheap even for large files.
    Returns ``None`` when the file holds no shots.
    """
    with zipfile.ZipFile(path) as archive:
        with archive.open("value_maps.npy") as handle:
            major, _ = np.lib.format.read_magic(handle)
            if major == 1:
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
    if len(shape) != 3 or shape[0] == 0:
        return None
    return int(shape[1]), int(shape[2])


def check_shot_map_files(
    paths: Iterable[Path | str],
    expected_shape: tuple[int, int] = CURRENT_SHOT_MAP_SHAPE,
) -> dict[str, object]:
    """Check every shot-maps file against ``expected_shape``.

    Returns a summary suitable for a run manifest. Raises
    :class:`ShotMapFormatError` listing the offending shapes and an example
    file for each if any file disagrees.
    """
    expected = tuple(int(value) for value in expected_shape)
    by_shape: dict[tuple[int, int], list[str]] = {}
    empty = 0
    for path in paths:
        shape = read_value_map_shape(path)
        if shape is None:
            empty += 1
            continue
        by_shape.setdefault(shape, []).append(str(path))

    mismatched = {shape: files for shape, files in by_shape.items() if shape != expected}
    if mismatched:
        details = "; ".join(
            f"{len(files)} file(s) with shape {_describe(shape)}, e.g. {files[0]}"
            for shape, files in sorted(mismatched.items())
        )
        raise ShotMapFormatError(
            f"Shot maps must use the current xG grid {expected}: {details}. "
            "Replace these files with the current xG data before running."
        )
    return {
        "value_map_shape": list(expected),
        "files_checked": sum(len(files) for files in by_shape.values()),
        "empty_files": empty,
    }


def check_loaded_shot_maps(
    shot_maps: Mapping[int, Mapping[str, object]],
    expected_shape: tuple[int, int] = CURRENT_SHOT_MAP_SHAPE,
    *,
    source: Path | str | None = None,
) -> None:
    """Raise :class:`ShotMapFormatError` if any loaded value map has another shape."""
    expected = tuple(int(value) for value in expected_shape)
    shapes = {tuple(np.shape(entry["value_map"])) for entry in shot_maps.values()}
    unexpected = sorted(shape for shape in shapes if shape != expected)
    if unexpected:
        where = f" in {source}" if source is not None else ""
        found = ", ".join(_describe(shape) for shape in unexpected)
        raise ShotMapFormatError(
            f"Shot maps{where} have value-map shape {found}; expected the current xG grid {expected}. "
            "Replace these files with the current xG data before running."
        )


def player_shot_map_paths(data_dir: Path | str, player_id: int, seasons: Iterable[int]) -> list[Path]:
    """Existing shot-maps files for one player and the requested seasons."""
    data_subdir = Path(data_dir) / "players" / f"player_{int(player_id)}" / "data"
    paths = (data_subdir / f"shot_maps_{int(season)}.npz" for season in seasons)
    return [path for path in paths if path.exists()]


__all__ = [
    "CURRENT_SHOT_MAP_SHAPE",
    "LEGACY_SHOT_MAP_SHAPE",
    "ShotMapFormatError",
    "check_loaded_shot_maps",
    "check_shot_map_files",
    "player_shot_map_paths",
    "read_value_map_shape",
]
