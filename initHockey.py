import os
from datetime import datetime

"""
This script initializes the directory structure for the hockey-multi domain experiments.
It creates folders for storing experiment data and plots based on the current date.
"""

def main():
    domain = "hockey-multi"
    now = datetime.now()

    base_experiment_path = os.path.join("Experiments", domain, f"{now.year}-{now.month}-{now.day}")
    plots_folder = os.path.join(base_experiment_path, "Data", "Plots")
    os.makedirs(plots_folder, exist_ok=True)


if __name__ == "__main__":
    main()
