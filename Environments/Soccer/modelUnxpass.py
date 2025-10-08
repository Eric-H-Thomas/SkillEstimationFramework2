import sys, code
sys.path.insert(0, '../un-xPass')

from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import mlflow
from xgboost import XGBClassifier


from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset
from unxpass.components import pass_success
from unxpass.components.utils import log_model, load_model
from unxpass.visualization import plot_action

import unxpass


def main():

	# How to train a pass success model using XGBoost


	# LOAD THE DATASET
	STORES_FP = Path("../un-xPass/stores")

	db = SQLiteDatabase(STORES_FP / "database.sqlite", ":memory:")

	code.interact("...", local=dict(globals(), **locals()))


	dataset_train = partial(PassesDataset, path=STORES_FP / "datasets" / "default" / "train")
	dataset_test = partial(PassesDataset, path=STORES_FP / "datasets" / "default" / "test")



	# CONFIGURE THE MODEL
	model = pass_success.XGBoostComponent(
    model=XGBClassifier(
        objective="binary:logistic", 
        eval_metric="auc"
        # you probably want to do some hyperparameter tuning here to get a good model
    ),
    features={
        'startpolar': [
            'start_dist_to_goal_a0',
            'start_angle_to_goal_a0'
        ],
        'relative_startlocation': [
            'start_dist_goalline_a0',
            'start_dist_sideline_a0'
        ],
        'endpolar': [
            'end_dist_to_goal_a0',
            'end_angle_to_goal_a0'
        ],
        'relative_endlocation': [
            'end_dist_goalline_a0',
            'end_dist_sideline_a0'
        ],
        'movement': [
            'movement_a0',
            'dx_a0',
            'dy_a0'
        ],
        'angle': [
            'angle_a0'
        ],
        'ball_height_onehot': [
            'ball_height_ground_a0',
            'ball_height_low_a0',
            'ball_height_high_a0'
        ],
        'player_possession_time': [
            'player_possession_time_a0'
        ],
        'speed': [
            'speed_a01',
            'speed_a02'
        ],
        'under_pressure': [
            'under_pressure_a0'
        ],
        'dist_defender': [
            'dist_defender_start_a0',
            'dist_defender_end_a0',
            'dist_defender_action_a0'
        ],
        'nb_opp_in_path': [
            'nb_opp_in_path_a0'
        ]
   			}, 
	)


	# TRAIN AND TEST THE MODEL
	model.train(dataset_train)

	# You can now log the model in the MLFflow registry
	mlflow.set_experiment(experiment_name="pass_success/threesixty")
	modelinfo = log_model(model, artifact_path="component")
	print(f"Model saved as {modelinfo.model_uri}")


	# Then you can reload it later
	model = load_model("runs:/1353b9d8d56e4cdc843719f70aab0c4c/component")


	# Next, evaluate how the model performs on a test set
	model.test(dataset_test)

	p_success = model.predict(dataset_test)
	print(p_success)


	code.interact("...", local=dict(globals(), **locals()))

	# Visualize what a pass what a high and a low success probability look like.
	easy_pass, hard_pass = (3795506, 4), (3795506, 2791)
	df_actions = db.actions(game_id=3795506)

	fig, ax = plt.subplots(1, 2, figsize=(12,4))
	plot_action(df_actions.loc[easy_pass], ax=ax[0])
	ax[0].set_title(f"P(success) = {p_success.loc[easy_pass]:.2f}")
	plot_action(df_actions.loc[hard_pass], ax=ax[1])
	ax[1].set_title(f"P(success) = {p_success.loc[hard_pass]:.2f}")
	plt.show()


	# Estimate the success probability of a pass towards every other location on the pitch.
	p_success_surfaces = model.predict_surface(dataset_test, game_id=3795506, db=db, x_bins=52, y_bins=34)

	df_actions = db.actions(game_id=3795506)
	# game_id, action_id
	sample = (3795506, 4)

	fig, ax = plt.subplots(1, 1, figsize=(6,4))
	plot_action(df_actions.loc[sample], surface=p_success_surfaces[f"action_{sample[1]}"], ax=ax, surface_kwargs={"cmap": "magma", "vmin": 0, "vmax": 1, "interpolation": "bilinear"})
	plt.show()

	db.close()


if __name__ == '__main__':
	main()

