from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

from launchexp.metrics import amae

NOMINAL_CLASSIFIERS = [
    "logisticregressor",
    "mlpclassifier",
    "ridgeclassifier",
    # "randomforestclassifier",
    # "xgboostclassifier",
    # "rotationforestclassifier",
    # "lightgbmclassifier",
    # "rvflnclassifier",
]

ORDINAL_CLASSIFIERS = [
    "logisticat",
    "logisticit",
    # "logisticat_no_cw",
    # "logisticit_no_cw",
    "mlpclmclassifier",
]


def set_estimator(
    estimator_name,
    base_grid,
    param_grid,
    random_state,
    interactive,
    n_jobs=-1,
    **kwargs,
):
    estimator_name = estimator_name.casefold()

    if estimator_name in NOMINAL_CLASSIFIERS:

        match estimator_name:

            case "logisticregressor":
                from sklearn.linear_model import LogisticRegression

                estimator = LogisticRegression(**base_grid)

            case "ridgeclassifier":
                from sklearn.linear_model import RidgeClassifier

                estimator = RidgeClassifier(**base_grid)

            case "svm":
                from sklearn.svm import SVC

                estimator = SVC(**base_grid)

            case "mlpclassifier":
                from launchexp.classification import MLPClassifier

                estimator = MLPClassifier(**base_grid)

            case "randomforestclassifier":
                from sklearn.ensemble import RandomForestClassifier

                estimator = RandomForestClassifier(**base_grid)

            case "xgboostclassifier":
                from xgboost import XGBClassifier

                estimator = XGBClassifier(**base_grid)

            case "rotationforestclassifier":
                from launchexp.classification import RotationForestClassifier

                estimator = RotationForestClassifier(**base_grid)

            case "lightgbmclassifier":
                from lightgbm import LGBMClassifier

                estimator = LGBMClassifier(**base_grid)

            case "rvflnclassifier":
                from rvfln.bls import BLSClassifier

                estimator = BLSClassifier(**base_grid)

            case _:
                raise NotImplementedError(
                    f"Estimator {estimator_name} was included in NOMINAL_CLASSIFIERS "
                    + "but not implemented in set_estimator function."
                )

    elif estimator_name in ORDINAL_CLASSIFIERS:

        match estimator_name:

            case "logisticat":
                from launchexp.ordinal_classification import LogisticAT

                estimator = LogisticAT(**base_grid)

            case "logisticit":
                from launchexp.ordinal_classification import LogisticIT

                estimator = LogisticIT(**base_grid)

            case "logisticat_no_cw":
                from mord import LogisticAT

                estimator = LogisticAT(**base_grid)

            case "logisticit_no_cw":
                from mord import LogisticIT

                estimator = LogisticIT(**base_grid)

            case "mlpbetaclassifier":
                from launchexp.ordinal_classification import MLPBetaClassifier

                estimator = MLPBetaClassifier(**base_grid)

            case "mlptriangularclassifier":
                from launchexp.ordinal_classification import MLPTriangularClassifier

                estimator = MLPTriangularClassifier(**base_grid)

            case "mlpclmclassifier":
                from launchexp.ordinal_classification import MLPCLMClassifier

                estimator = MLPCLMClassifier(**base_grid)

            case _:
                raise NotImplementedError(
                    f"Estimator {estimator_name} was included in ORDINAL_CLASSIFIERS "
                    + "but not implemented in set_estimator function."
                )

    else:
        raise ValueError(f"Estimator {estimator_name} not recognised.")

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        scoring=make_scorer(amae, greater_is_better=False),  # neg_median_absolute_error
        n_iter=30 if not interactive else 5,
        n_jobs=n_jobs,
        cv=3,
        error_score="raise",
        random_state=random_state,
        **kwargs,
    )
