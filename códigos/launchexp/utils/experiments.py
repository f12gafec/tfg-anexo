# -*- coding: utf-8 -*-
import json
import time
import os

import numpy as np
import pandas as pd
from dlordinal.metrics import minimum_sensitivity
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    recall_score,
    confusion_matrix,
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
    interactive=False
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
    conf_matrix = confusion_matrix(y_test, test_preds)

    # 1. Preparar datos planos para Excel
    flattened_data = {
        "dataset": dataset,
        "estimator": estimator_name,
        "seed": random_state,
        "total_time_ms": total_time
    }
    
    # Añadir métricas prefijadas
    for k, v in train_metrics.items():
        flattened_data[f"train_{k}"] = v
    for k, v in test_metrics.items():
        flattened_data[f"test_{k}"] = v
        
    # Añadir parámetros
    best_params = estimator.best_params_ if hasattr(estimator, "best_params_") else None
    flattened_data["best_params"] = str(best_params)

    # 2. Guardar como DataFrame en Excel (Nombre único para no sobrescribir)
    os.makedirs(results_dir, exist_ok=True)
    file_name = f"summary_{dataset}_{estimator_name}_seed_{random_state}.xlsx"
    excel_path = os.path.join(results_dir, file_name)
    
    summary_df = pd.DataFrame([flattened_data])
    summary_df.to_excel(excel_path, index=False)
    print(f"Resumen guardado en Excel: {excel_path}")

    # 3. GUARDAR LA MATRIZ DE CONFUSIÓN EN FORMATO NUMPY
    matrix_file_name = f"conf_matrix_{dataset}_{estimator_name}_seed_{random_state}.npy"
    matrix_path = os.path.join(results_dir, matrix_file_name)
    np.save(matrix_path, conf_matrix)
    print(f"Matriz de confusión guardada en: {matrix_path}")

    print("Train metrics")
    print(json.dumps(train_metrics, indent=4))
    print("Test metrics")
    print(json.dumps(test_metrics, indent=4))
    print("Matriz confusión")
    print(conf_matrix)


def load_data(data_dir, dataset, random_state, interactive):
    inputs = dataset.split("_")

    if inputs[1] == "sensors":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
        
        # 1h prediction task: Leemos las columnas por SU NOMBRE. 
        # El [:-1] coge todos los datos menos la última fila para predecir el futuro.
        hour = df['hour'].values[:-1].astype(float)
        wind_dir = df['wind_dir'].values[:-1].astype(float)
        wind_speed = df['wind_speed'].values[:-1].astype(float)
        temp = df['temp'].values[:-1].astype(float)
        dewpt = df['dewpt'].values[:-1].astype(float)
        press = df['press'].values[:-1].astype(float)

        # --- FEATURE ENGINEERING ---
        
        # 1. HORA CÍCLICA
        hour_rads = 2 * np.pi * hour / 24.0
        hour_sin = np.sin(hour_rads)
        hour_cos = np.cos(hour_rads)

        # 2. DIRECCIÓN DEL VIENTO CÍCLICA
        wind_dir_sin = np.sin(np.deg2rad(wind_dir))
        wind_dir_cos = np.cos(np.deg2rad(wind_dir))

        # 3. HUMEDAD RELATIVA (Magnus-Tetens)
        num = np.exp((17.625 * dewpt) / (243.04 + dewpt))
        den = np.exp((17.625 * temp) / (243.04 + temp))
        rh = 100 * (num / den)

        # Ensamblamos la matriz X final con nuestras variables calculadas
        X = np.column_stack((wind_speed, temp, dewpt, press, wind_dir_sin, wind_dir_cos, hour_cos, hour_sin, rh))

        # Definimos los nombres de las columnas en el mismo orden que el stack
        feature_names = ['WS', 'Temp', 'Dewpt', 'Press', 'WD_Sin', 'WD_Cos']

        # Imprimimos un encabezado y las primeras 5 filas
        print("\n--- Variables de entrada (X) ---")
        print(f"{' | '.join(feature_names)}")
        print("-" * 50)
        print(X[:5]) # Muestra las primeras 5 filas
        print(f"Forma de la matriz X: {X.shape}\n")

    elif inputs[1] == "era5":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_era5.csv", header=0)
        X = df.values[:-1, 5:].astype(float)

    elif inputs[1] == "concat":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
        
        # Para concat, hacemos lo mismo seleccionando las columnas exactas
        hour = df['hour'].values[:-1].astype(float)
        wind_dir = df['wind_dir'].values[:-1].astype(float)
        wind_speed = df['wind_speed'].values[:-1].astype(float)
        temp = df['temp'].values[:-1].astype(float)
        dewpt = df['dewpt'].values[:-1].astype(float)

        # Cálculos físicos
        hour_rads = 2 * np.pi * hour / 24.0
        hour_sin = np.sin(hour_rads)
        hour_cos = np.cos(hour_rads)

        wind_dir_sin = np.sin(np.deg2rad(wind_dir))
        wind_dir_cos = np.cos(np.deg2rad(wind_dir))

        num = np.exp((17.625 * dewpt) / (243.04 + dewpt))
        den = np.exp((17.625 * temp) / (243.04 + temp))
        rh = 100 * (num / den)

        X_sensors = np.column_stack((wind_speed, temp, dewpt, hour_sin, hour_cos, wind_dir_sin, wind_dir_cos, rh))

        df_2 = pd.read_csv(f"{data_dir}/{inputs[0]}_era5.csv", header=0)
        X = np.column_stack((X_sensors, df_2.values[:-1, 7:].astype(float)))

    # Extraemos la variable objetivo 'vis' desplazada 1 hora [1:]
    df_output = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
    y = df_output['vis'].values[1:]

    if inputs[2] == "autoreg":
        df = pd.read_csv(f"{data_dir}/{inputs[0]}_sensors.csv", header=0)
        # Añadimos la visibilidad de la hora actual como predictor
        #X = np.column_stack((X, df['vis'].values[:-1].astype(float)))
        X = df['vis'].values[:-1].reshape(-1, 1).astype(float)

    # Discretizamos el target en 4 clases
    y = pd.cut(y, bins=[-np.inf, 500, 1000, 5000, np.inf], labels=[0, 1, 2, 3])

    print(
        "Full (100%) ->",
        np.unique(y, return_counts=True)[1],
        np.round(np.unique(y, return_counts=True)[1] / len(y) * 100, 2),
    )

    # Separamos en train y test cronológicamente
    test_starts = int(len(y) * 0.75)

    X_train, y_train = X[:test_starts], y[:test_starts]
    X_test, y_test = X[test_starts:], y[test_starts:]

    # Escalado de datos
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



