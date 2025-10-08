from unxpass.visualization import plot_action
from mplsoccer import Pitch

import numpy as np
import sys, os
import code

import matplotlib.pyplot as plt


if not os.path.exists(f"Environments{os.sep}Soccer{os.sep}Plots"):
    os.mkdir(f"Environments{os.sep}Soccer{os.sep}Plots")


if len(sys.argv) != 3:
    print("Please provide the x_bins & y_bins params.")
    exit()

x_bins = int(sys.argv[1])
y_bins = int(sys.argv[2])

print(f"x_bins: {x_bins}")
print(f"y_bins: {y_bins}")


_spadl_cfg = {
    "length": 105,
    "width": 68,
    "penalty_box_length": 16.5,
    "penalty_box_width": 40.3,
    "six_yard_box_length": 5.5,
    "six_yard_box_width": 18.3,
    "goal_width": 7.32,
    "penalty_spot_distance": 11,
    "goal_length": 2,
    "origin_x": 0,
    "origin_y": 0,
    "circle_radius": 9.15,
}

yy, xx = np.ogrid[0.5:y_bins, 0.5:x_bins]

# - map to spadl coordinates
x_coo = np.clip(xx / x_bins * _spadl_cfg["length"], 0, _spadl_cfg["length"])
y_coo = np.clip(yy / y_bins * _spadl_cfg["width"], 0, _spadl_cfg["width"])

actions = np.array(np.meshgrid(x_coo, y_coo)).T.reshape(-1, 2)

p = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68)
_, ax = p.draw(figsize=(12, 8))

p.scatter(actions[:,0],actions[:,1], c="#6CABDD", s=80, ec="k", ax=ax)

plt.savefig(f"Environments{os.sep}Soccer{os.sep}Plots{os.sep}xb-{x_bins}-yb-{y_bins}.png", bbox_inches='tight')
plt.clf()  

# code.interact("...", local=dict(globals(), **locals()))


