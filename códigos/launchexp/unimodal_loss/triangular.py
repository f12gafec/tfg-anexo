from math import sqrt

import numpy as np
import torch
from dlmisc.losses import UnimodalCrossEntropyLoss


def triangular_pdf(x: float, a: float, b: float, c: float):
    """
    Triangular distribution PDF.
    """
    if x < a:
        return 0
    if a <= x < c:
        return (2 * (x - a)) / ((b - a) * (c - a))
    if x == c:
        return 2 / (b - a)
    if c < x <= b:
        return (2 * (b - x)) / ((b - a) * (b - c))
    if b < x:
        return 0


def triangular_cdf(x: float, a: float, b: float, c: float):
    """
    Triangular distribution CDF.
    """
    if x <= a:
        return 0.0
    if a < x < c:
        return pow(x - a, 2) / ((b - a) * (c - a))
    if c < x < b:
        return 1.0 - pow(b - x, 2) / ((b - a) * (b - c))
    if b <= x:
        return 1.0

    raise ValueError(
        "Triangular CDF could not be computed. Check the parameters and the input value."
    )


def get_triangular_parameters(j, J, alphas, thresholds):
    if len(thresholds) != J + 1:
        raise ValueError(
            f"The number of thresholds must be equal to the number of classes + 1 ({len(thresholds)=}, {J=})."
        )

    if len(alphas) != J:
        raise ValueError("The number of alphas must be equal to the number of classes.")

    if j < 0 or j >= J:
        raise ValueError("The level index must be between 0 and J-1.")

    if j == 0:
        b = thresholds[1] / (1 - sqrt(alphas[0]))
        a = 0
        c = 0
    elif j == J - 1:
        a = (thresholds[j] - sqrt(alphas[j])) / (1 - sqrt(alphas[j]))
        b = 1
        c = 1
    else:
        b_num = thresholds[j] * sqrt(alphas[j] / 2.0) + thresholds[j + 1] * (
            sqrt(alphas[j] / 2.0) - 1.0
        )
        b_denom = 2 * sqrt(alphas[j] / 2.0) - 1.0
        b = b_num / b_denom
        a = thresholds[j] + thresholds[j + 1] - b
        c = (a + b) / 2.0

    return a, b, c


def get_asymmetric_triangular_parameters(j, J, alphas, thresholds):
    if len(thresholds) != J + 1:
        raise ValueError(
            f"The number of thresholds must be equal to the number of classes + 1 ({len(thresholds)=}, {J=})."
        )

    if len(alphas) != 2 * J:
        raise ValueError("The number of alphas must be equal to the number of classes.")

    if j < 0 or j >= J:
        raise ValueError("The level index must be between 0 and J-1.")

    if j == 0:
        b = thresholds[1] / (1 - sqrt(alphas[0]))
        a = 0
        c = 0
    elif j == J - 1:
        a = (thresholds[j] - sqrt(alphas[j])) / (1 - sqrt(alphas[j]))
        b = 1
        c = 1
    else:
        from scipy.optimize import fsolve

        c = (thresholds[j - 1] + thresholds[j]) / 2.0

        def f(a):
            def b():
                return a - (thresholds[j - 1] - a) ** 2 / (alphas[j - 1] * (c - a))

            a = b() - (b() - thresholds[j]) ** 2 / (alphas[j + 3] * (b() - c))
            return a

        a = fsolve(f, 0.5)[0]
        b = a - (thresholds[j - 1] - a) ** 2 / (alphas[j - 1] * (c - a))

    return a, b, c


def get_triangular_probabilities(num_classes, alphas, thresholds):
    probs = []

    for j in range(num_classes):
        j_probs = []
        a, b, c = get_triangular_parameters(j, num_classes, alphas, thresholds)
        for k in range(num_classes):
            j_probs.append(
                triangular_cdf(thresholds[k + 1], a, b, c)
                - triangular_cdf(thresholds[k], a, b, c)
            )

        probs.append(j_probs)

    return np.array(probs)


class TriangularCrossEntropyLoss(UnimodalCrossEntropyLoss):
    def __init__(
        self,
        num_classes: int = 5,
        alphas: list[float] = [0.1, 0.1, 0.1, 0.1, 0.1],
        thresholds: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(num_classes, eta=0.85, **kwargs)

        if thresholds is None:
            from .utils import get_intervals

            thresholds = get_intervals(num_classes)

        self.cls_probs = torch.tensor(
            get_triangular_probabilities(num_classes, alphas, thresholds)
        )
