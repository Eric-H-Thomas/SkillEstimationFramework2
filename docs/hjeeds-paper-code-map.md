# Paper-to-code map

Section titles and LaTeX labels are used because page and section numbers change
between manuscript versions.

| Paper location | Primary implementation |
|---|---|
| Main, "Background on JEEDS" (`subsec:background_jeeds`) | `HJEEDS/likelihood.py`, `HJEEDS/estimation.py` |
| Main, "Population-Level Prior" (`subsec:population_prior`) | `HJEEDS/models.py`, `HJEEDS/sampling.py` |
| Main, "Individual-Level Likelihood" (`subsec:individual_likelihood`) | `HJEEDS/likelihood.py`, `HJEEDS/environment_adapters.py` |
| Main, "Inference Procedure" (`subsec:inference`) | `HJEEDS/estimation.py`, `HJEEDS/pipeline.py` |
| Main, "1D-Darts Environment" and "Experimental Procedure" | `HJEEDS/darts_environment.py`, `HJEEDS/darts_hierarchical_vs_jeeds.py` |
| Main, "Evaluation Metrics"; Supplement, "Percentage-Rationality Metric" | `HJEEDS/rationality.py` |
| Main, "Baseline Results" | `HJEEDS/darts_hierarchical_vs_jeeds.py`, `HJEEDS/plot_main_paper_baseline.py` |
| Main/Supplement, "Hyperprior Robustness" | `HJEEDS/darts_hierarchical_prior_sensitivity.py`, `HJEEDS/plot_hyperprior_robustness_appendix.py` |
| Main/Supplement, "Non-Gaussian Population Shape Sensitivity" | `HJEEDS/population_shapes.py`, `HJEEDS/darts_population_shape_sensitivity.py`, `HJEEDS/plot_population_shape_*.py` |
| Supplement, "Agents Per Observation-Count Bucket" | `HJEEDS/darts_agents_per_bucket_sensitivity.py`, `HJEEDS/agents_per_bucket_plot_data.py`, `HJEEDS/plot_agents_per_bucket_sensitivity_panels.py` |
| Supplement, "Outlier Contamination Sensitivity" | `HJEEDS/darts_outlier_sensitivity.py`, `HJEEDS/plot_outlier_sensitivity.py` |
| Supplement, "High-data Anchor Availability" | `HJEEDS/darts_anchor_availability_sensitivity.py`, `HJEEDS/plot_anchor_availability_robustness.py` |
| Main/Supplement, "Decision-Model Misspecification" | `HJEEDS/decision_models.py`, `HJEEDS/darts_decision_model_sensitivity.py` |
| Supplement, "True Population Correlation Sensitivity" | `HJEEDS/darts_true_correlation_sensitivity.py` |
| Supplement, "Grid Resolution Sensitivity" | `HJEEDS/darts_grid_resolution_sensitivity.py` |
| Main/Supplement, "Compound Stress Test" | `HJEEDS/darts_compound_stress_sensitivity.py` |
| Main, "2D-Darts Extension"; Supplement, "2D-Darts Hyperprior Selection" | `Environments/Darts/RandomDarts/two_d_darts.py`, `HJEEDS/two_d_completion.py` |
| Main, "MLB Pitching"; Supplement, "MLB Hyperprior Selection" | `HJEEDS/baseball_pitch.py`, `HJEEDS/baseball_likelihood.py`, `HJEEDS/baseball_hyperpriors.py`, `HJEEDS/baseball_convergence*.py` |
| Supplement, "MLB Between-Group Execution-Skill Gaps" | `HJEEDS/baseball_bbip.py`, `HJEEDS/baseball_roster.py`, `HJEEDS/baseball_separability.py`, `HJEEDS/plot_main_paper_higher_dimensional.py` |

The exact figure-to-renderer mapping and commands are in
`docs/hjeeds-paper-readme.md` under "Render publication figures."

