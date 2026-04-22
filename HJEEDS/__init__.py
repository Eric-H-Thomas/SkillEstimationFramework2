# This file still requires human verification. Delete this comment when done.
"""Hierarchical JEEDS experiment package."""

# This package isolates the hierarchical darts experiment from the rest of the
# repository.  The individual modules are intentionally small and focused so a
# reviewer can inspect the experiment one concern at a time:
#
# - ``config`` defines the study configuration and command-line interface.
# - ``models`` holds the shared dataclasses used across the package.
# - ``sampling`` simulates the synthetic darts data.
# - ``likelihood`` computes the per-agent JEEDS likelihood grid.
# - ``estimation`` runs the independent and hierarchical inference routines.
# - ``aggregation`` and ``artifacts`` turn per-agent results into outputs.
# - ``pipeline`` and ``darts_hierarchical_vs_jeeds`` orchestrate execution.
