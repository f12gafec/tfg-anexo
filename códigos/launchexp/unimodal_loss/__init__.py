# -*- coding: utf-8 -*-
"""Unimodal regularised loss functions."""

from .beta import BetaCrossEntropyLoss, get_beta_params
from .betagen import BetaGenCrossEntropyLoss
from .general_triangular import (
    GeneralTriangularCrossEntropyLoss,
    get_general_triangular_params,
    get_general_triangular_probabilities,
)
from .triangular import (
    TriangularCrossEntropyLoss,
    get_triangular_parameters,
    get_triangular_probabilities,
    triangular_cdf,
    triangular_pdf,
)

__all__ = [
    "BetaCrossEntropyLoss",
    "BetaGenCrossEntropyLoss",
    "get_beta_params",
    "get_triangular_parameters",
    "get_triangular_probabilities",
    "get_general_triangular_params",
    "get_general_triangular_probabilities",
    "triangular_pdf",
    "triangular_cdf",
    "TriangularCrossEntropyLoss",
    "GeneralTriangularCrossEntropyLoss",
]
