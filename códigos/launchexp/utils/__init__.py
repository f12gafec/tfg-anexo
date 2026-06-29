# -*- coding: utf-8 -*-
"""Experiment functions."""

__all__ = [
    "load_and_run_experiment",
    "load_data",
    "set_estimator",
    "compute_metrics",
    "NOMINAL_CLASSIFIERS",
    "ORDINAL_CLASSIFIERS",
]

from launchexp.utils.experiments import (
    compute_metrics,
    load_and_run_experiment,
    load_data,
)
from launchexp.utils.set_estimator import (
    NOMINAL_CLASSIFIERS,
    ORDINAL_CLASSIFIERS,
    set_estimator,
)
