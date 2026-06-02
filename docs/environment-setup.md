# Environment Setup

<!-- This file was written or edited by AI and still requires human review. Delete this comment when done -->

Follow these steps to prepare a working environment for the Skill Estimation Framework.

## Python version
- Use Python 3.10.6 (also tested with 3.8.3 and 3.9.10).

## Required libraries (conda install commands)
- numpy  (`conda install -c anaconda numpy`)
- scipy  (`conda install -c anaconda scipy`)
- scikit_learn (`conda install -c anaconda scikit-learn`)
- pandas (`conda install -c anaconda pandas`) (version < 2.0)
- torch (`conda install -c pytorch pytorch`)
- psycopg2  (`conda install -c anaconda psycopg2`)
- matplotlib (`conda install -c conda-forge matplotlib`)
- chart_studio (`conda install -c plotly chart-studio`)
- tqdm (`conda install -c conda-forge tqdm`)
- jinja2 (`conda install -c anaconda jinja2`)
- pympler (for testing) (`conda install -c conda-forge pympler`)
- memory_profiler (`conda install -c conda-forge memory_profiler`)
- pybaseball (`pip install pybaseball`)
- multiprocess  (`conda install multiprocess`)
- openpyxl (`pip install openpyxl`)

## Repository checkout
1. Clone the `skill-estimation-framework` repository from GitHub.
2. Navigate into the cloned `skill-estimation-framework` directory.

## H-JEEDS notes
- H-JEEDS uses the same core scientific Python stack: `numpy`, `scipy`, and `matplotlib`.
- Cluster runs should set `MPLBACKEND=Agg`; the provided H-JEEDS Slurm helper does this automatically.
- Run H-JEEDS commands from the repository root so package imports such as `HJEEDS.darts_hierarchical_vs_jeeds` resolve correctly.

## Minimal H-JEEDS paper environment

For the H-JEEDS paper experiment runners, use the smaller environment file at the repository root:

```bash
module load miniforge3/25.3.1-0
eval "$(conda shell.bash hook)"
conda env create -f environment-hjeeds.yml
conda activate hjeeds-paper
```

Confirm the required scientific stack is available:

```bash
python -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)"
```

Then submit Slurm runs with the active environment's Python:

```bash
python run_hjeeds_paper_experiments.py \
  --mode slurm \
  --seed 12345 \
  --num-seeds 3 \
  --output-root HJEEDS/results/hjeeds_paper_3_seed_slurm_check \
  --python-bin "$(which python)" \
  --qos normal \
  --time 23:00:00 \
  --mem 16G
```

If a conda environment is not available, `requirements-hjeeds.txt` contains the same minimal package set for a pip-based virtual environment.
