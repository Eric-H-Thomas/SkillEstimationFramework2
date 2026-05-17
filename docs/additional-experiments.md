# Additional Experiments

<!-- This file was written or edited by AI and still requires human review. Delete this comment when done -->

The repository includes scripts that study how the JEEDS estimator reacts to suboptimal aiming in the darts domain.

- Supports sharded local runs, Slurm submissions, and a verification harness for the parallel path.
- See [Testing/jeeds_aiming_sensitivity.md](../Testing/jeeds_aiming_sensitivity.md) for the full workflow and usage examples.

The repository also includes H-JEEDS ablation scripts for the hierarchical 1D darts estimator.

- The standalone hyperprior robustness study runs a 60-condition grid covering average skill, population spread, correlation, and combined misspecification.
- The agents-per-bucket ablation defaults to 15 scenarios: five agents-per-bucket values crossed with three representative robustness conditions.
- See [H-JEEDS Experiments](hjeeds.md) for commands, Slurm submission, outputs, and verification checks.
