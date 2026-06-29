import torch
from dlmisc.distributions.beta import beta_dist
from dlmisc.losses import UnimodalCrossEntropyLoss


def get_beta_probabilities(num_classes, u, v, thresholds):
    """Get the probabilities of the beta distribution for each class.
    For each, the probability is computed as the integral of the beta
    distribution between the corresponding thresholds.

    Parameters
    ----------
    num_classes : int
        Number of classes.
    u : int
        First parameter of the beta distribution.
    v : int
        Second parameter of the beta distribution.
    thresholds : list
        List of thresholds.

    Returns
    -------
    list
        List of probabilities.
    """

    probs = []
    for j in range(num_classes):
        bj = thresholds[j]
        bj1 = thresholds[j + 1]
        probs.append(
            beta_dist(thresholds[j + 1], u, v) - beta_dist(thresholds[j], u, v)
        )

    return probs


def get_beta_params(num_classes, thresholds, j):
    if len(thresholds) != num_classes:
        raise ValueError(
            f"Number of thresholds ({len(thresholds)}) must be equal "
            + f"to the number of classes ({num_classes})"
        )

    from math import sqrt

    from scipy.optimize import fsolve

    bj = thresholds[j]
    bj1 = thresholds[j + 1]
    v_u = lambda u: (2 * u) / (bj1 - bj) - u

    ex = lambda u: u / (u + v_u(u))
    sx = lambda u: sqrt((u * v_u(u)) / ((u + v_u(u)) ** 2 * (u + v_u(u) + 1)))

    eq1 = lambda u: ex(u) - sx(u) - bj
    eq2 = lambda u: ex(u) + sx(u) - bj1
    eq3 = lambda u: ex(u) - sx(u) - bj1
    eq4 = lambda u: ex(u) + sx(u) - bj

    try:
        u = fsolve(eq1, 0.01)
        v = v_u(u)
    except:
        try:
            u = fsolve(eq2, 0.01)
            v = v_u(u)
        except:
            try:
                u = fsolve(eq3, 0.01)
                v = v_u(u)
            except:
                try:
                    u = fsolve(eq4, 0.01)
                    v = v_u(u)
                except:
                    raise ValueError("No solution found")

    return u, v


class BetaCrossEntropyLoss(UnimodalCrossEntropyLoss):
    """Beta unimodal regularised cross entropy loss."""

    def __init__(
        self, num_classes: int = 5, eta: float = 1.0, thresholds=None, **kwargs
    ):
        """
        Parameters
        ----------
        num_classes : int, default=5
            Number of classes.
        eta : float, default=1.0
            Parameter that controls the influence of the regularisation.
        thresholds : list, default=None
            List of thresholds. If None, the intervals equally spaced
            between 0 and 1 are used.
        """

        super().__init__(num_classes, eta, **kwargs)

        if thresholds is None:
            from .utils import get_intervals

            thresholds = get_intervals(num_classes)

        # Precompute class probabilities for each label
        self.cls_probs = torch.tensor(
            [
                get_beta_probabilities(
                    num_classes,
                    *get_beta_params(
                        num_classes=num_classes, thresholds=thresholds, j=i
                    ),
                    thresholds,
                )
                for i in range(num_classes)
            ]
        ).float()

        print(self.cls_probs)
