from itertools import combinations_with_replacement

from launchexp.utils.set_estimator import NOMINAL_CLASSIFIERS, ORDINAL_CLASSIFIERS


def set_params_grid(estimator_name, random_state):
    estimator_name = estimator_name.casefold()

    if estimator_name in NOMINAL_CLASSIFIERS:

        match estimator_name:

            case "logisticregressor":

                base_grid = {
                    "class_weight": "balanced",
                    "random_state": random_state,
                    "solver": "saga",
                }

                param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "ridgeclassifier":

                base_grid = {"class_weight": "balanced", "random_state": random_state}

                param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "svm":

                base_grid = {
                    "probability": True,
                    "class_weight": "balanced",
                    "random_state": random_state,
                }

                param_grid = {
                    "kernel": ["rbf", "linear"],
                    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    "gamma": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                }

            case "mlpclassifier":

                base_grid = {"class_weight": "balanced", "batch_size": 1024}

                param_grid = {
                    "hidden_units": [
                        comb
                        for r in range(2, 5)
                        for comb in combinations_with_replacement(
                            [10, 50, 100, 250, 500], r
                        )
                    ],
                    "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
                    "max_iter": [250, 500, 750, 1000],
                }

            case "randomforestclassifier":

                base_grid = {"class_weight": "balanced", "random_state": random_state}

                param_grid = {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "max_features": [1.0, "sqrt"],
                    "min_samples_leaf": [1, 2, 4],
                    "min_samples_split": [2, 5, 10],
                }

            case "xgboostclassifier":

                base_grid = {"random_state": random_state}

                param_grid = {
                    "max_depth": [4, 6, 8],
                    "n_estimators": [100, 200, 500],
                    "subsample": [0.5, 0.75, 1],
                    "learning_rate": [0.1, 0.3, 0.5, 0.7],
                }


            case "rotationforestclassifier":
                from sklearn.tree import DecisionTreeClassifier
                
                # 1. Definimos el árbol base restringido
                base_tree = DecisionTreeClassifier(
                    criterion="entropy", 
                    random_state=random_state
                )

                # 2. Se lo pasamos al Rotation Forest
                base_grid = {
                    "random_state": random_state,
                    "base_estimator": base_tree
                }

                # 3. Ampliamos el grid para frenar el overfitting
                param_grid = {
                    # Bajamos un poco los estimadores para que no tarde días en buscar
                    "n_estimators": [100, 250, 500], 
                    "remove_proportion": [0.25, 0.5, 0.75],
                    
                    # Límite de profundidad (evita que el árbol memorice ruido)
                    "base_estimator__max_depth": [5, 10, 15], 
                    
                    # Mínimo de muestras por hoja (fuerza a generalizar)
                    "base_estimator__min_samples_leaf": [5, 10, 20] 
                }

            case "lightgbmclassifier":

                base_grid = {
                    "class_weight": "balanced",
                    "random_state": random_state,
                    "verbose": -1,
                    "n_jobs": 1,
                }

                param_grid = {
                    "n_estimators": [100, 500, 1000],
                    "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    "num_leaves": [10, 20, 30, 40, 50],
                    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                }

            case "rvflnclassifier":

                base_grid = {"n_z": 5, "n_z_features": 5, "n_h": 5}

                param_grid = {
                    "n_z": [5, 10, 15, 20, 25, 50, 100, 500],
                    "n_z_features": [5, 10, 15, 20, 25, 50, 100],
                    "n_h": [5, 10, 15, 20, 25, 50, 100, 500],
                    "alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                }

            case _:
                raise NotImplementedError(
                    f"Estimator {estimator_name} was included in NOMINAL_CLASSIFIERS "
                    + "but not implemented in set_params_grid function."
                )

    elif estimator_name in ORDINAL_CLASSIFIERS:

        match estimator_name:

            case "logisticat":

                base_grid = {"class_weight": "balanced"}
                # base_grid = {"class_weight": {0: 30, 1: 78.5, 2: 11, 3: 1.1}}

                param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "logisticit":

                base_grid = {"class_weight": "balanced"}

                param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "logisticat_no_cw":

                base_grid = {}

                param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "logisticit_no_cw":

                base_grid = {}

                param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

            case "mlpbetaclassifier":

                base_grid = {"class_weight": "balanced", "batch_size": 1024}

                param_grid = {
                    "hidden_units": [
                        comb
                        for r in range(2, 5)
                        for comb in combinations_with_replacement(
                            [10, 50, 100, 250, 500], r
                        )
                    ],
                    "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
                    "max_iter": [250, 500, 750, 1000],
                    "loss_eta": [0.6, 0.8, 1.0],
                }

            case "mlptriangularclassifier":

                base_grid = {"class_weight": "balanced", "batch_size": 1024}

                param_grid = {
                    "hidden_units": [
                        comb
                        for r in range(2, 5)
                        for comb in combinations_with_replacement(
                            [10, 50, 100, 250, 500], r
                        )
                    ],
                    "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
                    "max_iter": [250, 500, 750, 1000],
                    "loss_alpha2": [0.05, 0.1, 0.15],
                    "loss_eta": [0.6, 0.8, 1.0],
                }

            case "mlpclmclassifier":

                base_grid = {"class_weight": "balanced", "batch_size": 1024}

                param_grid = {
                    "hidden_units": [
                        comb
                        for r in range(2, 5)
                        for comb in combinations_with_replacement(
                            [10, 50, 100, 250, 500], r
                        )
                    ],
                    "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4],
                    "max_iter": [250, 500, 750, 1000],
                    "link_function": ["logit", "probit", "cloglog"],
                }

            case _:
                raise NotImplementedError(
                    f"Estimator {estimator_name} was included in ORDINAL_CLASSIFIERS "
                    + "but not implemented in set_param_grid function."
                )

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")

    return base_grid, param_grid
