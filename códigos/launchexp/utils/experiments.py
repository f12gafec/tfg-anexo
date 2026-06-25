# -*- coding: utf-8 -*-
import json
import time

import numpy as np
import pandas as pd
from dlordinal.metrics import minimum_sensitivity
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from launchexp.metrics import amae, mmae
from launchexp.utils.set_estimator import set_estimator
from launchexp.utils.set_params_grid import set_params_grid


def load_and_run_experiment(
    data_dir,
    results_dir,
    dataset,
    random_state=0,
    estimator_name="logisticregressor",
    n_jobs=-1,
    interactive=False,
):
    from os import environ
    from random import seed as random_seed

    # Fix seeds
    np.random.seed(random_state)
    random_seed(random_state)
    environ["PYTHONHASHSEED"] = str(random_state)
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    X_train, y_train, X_test, y_test = load_data(
        data_dir, dataset, random_state, interactive
    )

    base_grid, param_grid = set_params_grid(estimator_name, random_state)

    estimator = set_estimator(
        estimator_name, base_grid, param_grid, random_state, interactive, n_jobs
    )

    config = get_config(
        estimator, estimator_name, base_grid, param_grid, dataset, random_state
    )


    if estimator_name is None:
        estimator_name = type(estimator).__name__

    start = int(round(time.time() * 1000))
    estimator.fit(X_train, y_train)

    try:
        train_probs = estimator.predict_proba(X_train)
        train_preds = estimator.classes_[np.argmax(train_probs, axis=1)]

        test_probs = estimator.predict_proba(X_test)
        test_preds = estimator.classes_[np.argmax(test_probs, axis=1)]

    except Exception:
        train_probs = np.array([])
        train_preds = estimator.predict(X_train)

        test_probs = np.array([])
        test_preds = estimator.predict(X_test)

    total_time = int(round(time.time() * 1000)) - start

    config = get_config(
        estimator, estimator_name, base_grid, param_grid, dataset, random_state
    )

    
    train_metrics = compute_metrics(y_train, train_preds)
    test_metrics = compute_metrics(y_test, test_preds)

    print("Train metrics")
    print(json.dumps(train_metrics, indent=4))
    print("Test metrics")
    print(json.dumps(test_metrics, indent=4))


def load_data(data_dir, dataset, random_state, interactive):

    inputs = dataset.split("_")

    if inputs[1] == "sensors":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
        # 1h prediction task. X is the current time, y is the next time
        X = df.values[:-1, 5:10]  # wind_dir, wind_speed, temp, dewpt, press

    elif inputs[1] == "era5":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_era5.csv", header=0)  # era5
        X = df.values[:-1, 5:]

    elif inputs[1] == "concat":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)  # sensors
        X = df.values[:-1, 5:9]
        df_2 = pd.read_csv(f"{data_dir}/{inputs[0]}_era5.csv", header=0)  # era5
        X = np.column_stack((X, df_2.values[:-1, 5:]))

    df_output = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
    y = df_output.values[1:, -1]

    if inputs[2] == "autoreg":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
        X = np.column_stack((X, df.values[:-1, -1]))

    # discretise the target in 4 classes
    y = pd.cut(y, bins=[-np.inf, 500, 1000, 5000, np.inf], labels=[0, 1, 2, 3])

    print(
        "Full (100%) ->",
        np.unique(y, return_counts=True)[1],
        np.round(np.unique(y, return_counts=True)[1] / len(y) * 100, 2),
    )

    # split the data into train and test
    test_starts = int(len(y) * 0.75)

    X_train, y_train = X[:test_starts], y[:test_starts]
    X_test, y_test = X[test_starts:], y[test_starts:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    print(
        "Train (75%) ->",
        np.unique(y_train, return_counts=True)[1],
        np.round(np.unique(y_train, return_counts=True)[1] / len(y_train) * 100, 2),
    )

    print(
        "Test  (25%) ->",
        np.unique(y_test, return_counts=True)[1],
        np.round(np.unique(y_test, return_counts=True)[1] / len(y_test) * 100, 2),
    )

    if interactive:
        return X_train, y_train, X_test, y_test
        return X_train[:5000], y_train[:5000], X_test[:1000], y_test[:1000]
    else:
        return X_train, y_train, X_test, y_test


def get_config(estimator, estimator_name, base_grid, param_grid, dataset, random_state):
    estimator_params = estimator.get_params().copy()

    config = {}
    config["estimator_name"] = estimator_name
    config["dataset"] = dataset
    config["base_grid"] = base_grid
    config["param_grid"] = param_grid
    config["random_state"] = random_state
    config["scoring"] = (
        estimator_params["scoring"]
        if isinstance(estimator_params["scoring"], str)
        else estimator_params["scoring"]._score_func.__name__
    )

    return config


def compute_metrics(targets, predictions):

    metrics = {
        "QWK": cohen_kappa_score(targets, predictions, weights="quadratic"),
        "MAE": mean_absolute_error(targets, predictions),
        "CCR": accuracy_score(targets, predictions),
        "MZE": 1.0 - accuracy_score(targets, predictions),
        "MS": minimum_sensitivity(targets, predictions),
        "BA": balanced_accuracy_score(targets, predictions),
        "AMAE": amae(targets, predictions),
        "MMAE": mmae(targets, predictions),
    }

    # Compute sensitivities for each class
    sensitivities = np.array(recall_score(targets, predictions, average=None))

    for i, sens in enumerate(sensitivities):
        metrics[f"Sens{i}"] = sens

    return metrics
