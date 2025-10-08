import sys, code
sys.path.insert(0, '../un-xPass')

from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import mlflow
from scipy.ndimage import zoom

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset
from unxpass.components import pass_selection, pass_value, pass_success
from unxpass.components.utils import load_model
from unxpass.visualization import plot_action
from unxpass.ratings import CreativeDecisionRating


def main():

	plt_settings = {"cmap": "magma", "vmin": 0, "vmax": 1, "interpolation": "bilinear"}


	STORES_FP = Path("../un-xPass/stores")

	db = SQLiteDatabase(STORES_FP / "database.sql")

	dataset_test = partial(PassesDataset, path=STORES_FP / "datasets" / "euro2020" / "test")



	# Select an example pass
	SAMPLE = (3795506,4)


	# Show the selected example
	ex_action = db.actions(game_id=SAMPLE[0]).loc[SAMPLE]

	# Causes error - not needed?
	# display(ex_action.to_frame().T)

	# fig, ax = plt.subplots(figsize=(6,4))
	# ax = plot_action(ex_action, ax=ax)
	# plt.show()

	model_pass_value = load_model('runs:/cd61e503f78d41be83a2691fab22cc15/component',)
	info = model_pass_value.test(dataset_test)
	print(info)
	
	code.interact("...", local=dict(globals(), **locals()))


	p_value_surfaces = model_pass_value.predict_surface(dataset_test,SAMPLE[0],db=db)

	# df_actions = db.actions(SAMPLE)

	fig, ax = plt.subplots(1, 1, figsize=(6,4))
	plot_action(ex_action, surface=p_value_surfaces[f"action_{SAMPLE[1]}"], ax=ax, surface_kwargs={"cmap": "magma", "vmin": 0, "vmax": 1, "interpolation": "bilinear"})
	plt.show()

	


if __name__ == '__main__':
	main()

