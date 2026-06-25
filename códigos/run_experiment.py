# -*- coding: utf-8 -*-

import sys
from launchexp.utils import load_and_run_experiment

def run_experiment(args):
    """Mechanism for testing estimator."""
    data_dir = "./data/"
    results_dir = "./results_debug/"
    estimator_name = "logisticregressor"
    dataset = "LEST_sensors_autoreg"
    resample = 0
    n_jobs = -1
    interactive = False

    print(f"Local Run of {estimator_name} over {dataset}.")

    load_and_run_experiment(
        data_dir,
        results_dir,
        dataset,
        random_state=resample,
        estimator_name=estimator_name,
        n_jobs=n_jobs,
        interactive=interactive,
    )


if __name__ == "__main__":
    run_experiment(sys.argv)
