"""Temporary script to compare original vs coordinate-remapped heatmaps.

Previously we used zoom() to resize (72,120) -> (40,60), which preserved the
wrong coordinate mapping. The Blackhawks grid covers Y:[-5,5] Z:[0,6], but
SpacesHockey (default grid for getAngularHeatmap) expects Y:[-3,3] Z:[0,4]. 
This script shows the original, the old zoom, and the correctly-remapped version.

After the fix, getAngularHeatmap now accepts optional grid_y/grid_z params,
so the JEEDS and blackhawks_plots code can pass the native BH grid directly
and let the full Y:[-5,5] Z:[0,6] surface be displayed in angular space.
A fourth panel shows the native map with overlay lines marking the 
SpacesHockey window.
"""
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Rectangle

# Configuration
PLAYER_ID = 950161
DATA_DIR = Path("Data/Hockey")
OUTPUT_DIR = Path("Data/Hockey/debug_heatmaps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Blackhawks grid (from queries.py location_y/location_z ranges)
BH_Y = np.linspace(-5.0, 5.0, 120)
BH_Z = np.linspace(0.0,  6.0,  72)

# SpacesHockey target grid
SH_Y = np.linspace(-3.0, 3.0, 60)
SH_Z = np.linspace(0.0,  4.0, 40)
SH_ZZ, SH_YY = np.meshgrid(SH_Z, SH_Y, indexing='ij')
SH_QUERY_PTS = np.stack([SH_ZZ.ravel(), SH_YY.ravel()], axis=-1)


def remap(value_map):
    interp = RegularGridInterpolator(
        (BH_Z, BH_Y), value_map, method='linear', bounds_error=False, fill_value=0.0,
    )
    return interp(SH_QUERY_PTS).reshape(40, 60)


def old_zoom(value_map):
    return zoom(value_map, (40 / value_map.shape[0], 60 / value_map.shape[1]), order=1)


print(f"Loading shot maps for player {PLAYER_ID}...")
shot_maps = {}
for pkl_file in sorted(glob.glob(str(DATA_DIR / f"player_{PLAYER_ID}/data/shot_maps_*.pkl"))):
    with open(pkl_file, "rb") as f:
        shot_maps.update(pickle.load(f))
print(f"Loaded {len(shot_maps)} shot maps\n")

for event_id in list(shot_maps.keys())[:5]:
    native = shot_maps[event_id]["value_map"]
    zoomed = old_zoom(native)
    remapped = remap(native)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    vmin, vmax = np.nanmin(native), np.nanmax(native)

    # Panel 0: Native with real-world axis ticks
    im0 = axes[0].imshow(native, origin="lower", cmap="jet", aspect="auto",
                         extent=[-5, 5, 0, 6], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Native (72×120)  Y:[-5,5]  Z:[0,6]", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Y (ft)"); axes[0].set_ylabel("Z (ft)")
    axes[0].axvline(-3, color='white', lw=1.5, ls='--', alpha=0.8)
    axes[0].axvline( 3, color='white', lw=1.5, ls='--', alpha=0.8)
    axes[0].axhline( 4, color='white', lw=1.5, ls='--', alpha=0.8, label='SpacesHockey bounds')
    axes[0].legend(loc='upper right', fontsize=8)
    plt.colorbar(im0, ax=axes[0])

    # Panel 1: Old zoom — wrong coordinate assumption
    im1 = axes[1].imshow(zoomed, origin="lower", cmap="jet", aspect="auto",
                         extent=[-3, 3, 0, 4], vmin=vmin, vmax=vmax)
    axes[1].set_title(f"OLD zoom (40×60) — WRONG assumed coords", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Y (ft)"); axes[1].set_ylabel("Z (ft)")
    plt.colorbar(im1, ax=axes[1])

    # Panel 2: Correct remap
    im2 = axes[2].imshow(remapped, origin="lower", cmap="jet", aspect="auto",
                         extent=[-3, 3, 0, 4], vmin=vmin, vmax=vmax)
    axes[2].set_title(f"FIXED remap (40×60) — correct net region", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Y (ft)"); axes[2].set_ylabel("Z (ft)")
    plt.colorbar(im2, ax=axes[2])

    # Panel 3: Native again with overlay showing the SpacesHockey window
    im3 = axes[3].imshow(native, origin="lower", cmap="jet", aspect="auto",
                         extent=[-5, 5, 0, 6], vmin=vmin, vmax=vmax)
    axes[3].set_title(f"Native + SpacesHockey window overlay", fontsize=11, fontweight='bold')
    axes[3].set_xlabel("Y (ft)"); axes[3].set_ylabel("Z (ft)")
    # Draw rectangle for the SpacesHockey bounds
    rect = Rectangle((-3, 0), 6, 4, linewidth=2.5, edgecolor='cyan', 
                     facecolor='none', linestyle='--', label='SpacesHockey grid')
    axes[3].add_patch(rect)
    axes[3].legend(loc='upper right', fontsize=8)
    plt.colorbar(im3, ax=axes[3])

    plt.suptitle(f"Event {event_id}", fontsize=12)
    plt.tight_layout()

    out_path = OUTPUT_DIR / f"heatmap_comparison_event_{event_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

print(f"\nDone. Comparison plots saved to {OUTPUT_DIR}")
