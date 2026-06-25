import math

import torch
from dlmisc.losses import UnimodalCrossEntropyLoss
from scipy.special import beta, hyp2f1

from .utils import get_intervals

_beta_params_sets = {
    "standard": {
        3: [[1, 4, 1], [4, 4, 1], [4, 1, 1]],
        4: [[1, 6, 1], [6, 10, 1], [10, 6, 1], [6, 1, 1]],
        5: [[1, 8, 1], [6, 14, 1], [12, 12, 1], [14, 6, 1], [8, 1, 1]],
        6: [[1, 10, 1], [7, 20, 1], [15, 20, 1], [20, 15, 1], [20, 7, 1], [10, 1, 1]],
        7: [
            [1, 12, 1],
            [7, 26, 1],
            [16, 28, 1],
            [24, 24, 1],
            [28, 16, 1],
            [26, 7, 1],
            [12, 1, 1],
        ],
        8: [
            [1, 14, 1],
            [7, 31, 1],
            [17, 37, 1],
            [27, 35, 1],
            [35, 27, 1],
            [37, 17, 1],
            [31, 7, 1],
            [14, 1, 1],
        ],
        9: [
            [1, 16, 1],
            [8, 40, 1],
            [18, 47, 1],
            [30, 47, 1],
            [40, 40, 1],
            [47, 30, 1],
            [47, 18, 1],
            [40, 8, 1],
            [16, 1, 1],
        ],
        10: [
            [1, 18, 1],
            [8, 45, 1],
            [19, 57, 1],
            [32, 59, 1],
            [45, 55, 1],
            [55, 45, 1],
            [59, 32, 1],
            [57, 19, 1],
            [45, 8, 1],
            [18, 1, 1],
        ],
        11: [
            [1, 21, 1],
            [8, 51, 1],
            [20, 68, 1],
            [34, 73, 1],
            [48, 69, 1],
            [60, 60, 1],
            [69, 48, 1],
            [73, 34, 1],
            [68, 20, 1],
            [51, 8, 1],
            [21, 1, 1],
        ],
        12: [
            [1, 23, 1],
            [8, 56, 1],
            [20, 76, 1],
            [35, 85, 1],
            [51, 85, 1],
            [65, 77, 1],
            [77, 65, 1],
            [85, 51, 1],
            [85, 35, 1],
            [76, 20, 1],
            [56, 8, 1],
            [23, 1, 1],
        ],
        13: [
            [1, 25, 1],
            [8, 61, 1],
            [20, 84, 1],
            [36, 98, 1],
            [53, 100, 1],
            [70, 95, 1],
            [84, 84, 1],
            [95, 70, 1],
            [100, 53, 1],
            [98, 36, 1],
            [84, 20, 1],
            [61, 8, 1],
            [25, 1, 1],
        ],
        14: [
            [1, 27, 1],
            [2, 17, 1],
            [5, 23, 1],
            [9, 27, 1],
            [13, 28, 1],
            [18, 28, 1],
            [23, 27, 1],
            [27, 23, 1],
            [28, 18, 1],
            [28, 13, 1],
            [27, 9, 1],
            [23, 5, 1],
            [17, 2, 1],
            [27, 1, 1],
        ],
    }
}


def get_beta_params(J, j, distribution="standard", alpha=1.0, beta=1.0):
    j = j + 1
    a, p, q = 0, 0, 0
    if distribution == "gen_v":
        if j == 1:
            p = 1
            q = (
                -(2 * p + 5) * (1 + alpha**2)
                + math.sqrt(
                    (2 * p + 5) ** 2 * (1 + alpha**2) ** 2
                    - 4
                    * (1 + alpha**2)
                    * (p**2 + 5 * p + 6)
                    * (1 + alpha**2 - 2 * J * alpha**2)
                )
            ) / (2 * (1 + alpha**2))
            a = 0.5
        elif j == J:
            a = 0.5
            q = 0.5
            C = (1 + beta**2 * (2 * J - 1) ** 2) / (2 * J * beta**2 * (2 * J - 1))
            _a = 4 - 4 * C
            _b = 20 - 24 * C
            _c = 24 - 35 * C

            # print(f'{_a=}, {_b=}, {_c=}, {_b**2 - 4*_a*_c=}')

            p = (-_b + math.sqrt(_b**2 - 4 * _a * _c)) / (2 * _a)

            # p = math.floor((q * (1 + beta**2 * (2*J - 1)**2)) /
            #                (4*J*beta**2 - beta**2 - 1))
            # p = -1 + J + math.sqrt(4 * J**2 - 2 * J + 1) / 2.0
            # p = (-7 + 6 * J + math.sqrt(36 * J**2 - 4 * J + 9)) / 8
        else:
            p, q, a = _beta_params_sets["standard"][J][j - 1]
    else:
        p, q, a = _beta_params_sets[distribution][J][j - 1]

    return p, q, a


def beta_dist(x, p, q, a=1.0):
    return (x ** (a * p)) / (p * beta(p, q)) * hyp2f1(p, 1 - q, p + 1, x**a)


def get_beta_probabilities(n, p, q, a=1.0):
    intervals = get_intervals(n)
    probs = []

    # Compute probability for each interval (class) using the distribution function.
    for interval in intervals:
        probs.append(beta_dist(interval[1], p, q, a) - beta_dist(interval[0], p, q, a))

    return probs


class BetaGenCrossEntropyLoss(UnimodalCrossEntropyLoss):
    def __init__(
        self,
        num_classes: int = 5,
        params_set: str = "standard",
        eta: float = 1.0,
        alpha=1.0,
        beta=1.0,
        **kwargs
    ):
        super().__init__(num_classes, eta, **kwargs)
        # self.params = _beta_params_sets[params_set]

        # Precompute class probabilities for each label
        self.cls_probs = torch.tensor(
            [
                get_beta_probabilities(
                    num_classes,
                    *get_beta_params(num_classes, i, params_set, alpha, beta)
                )
                for i in range(num_classes)
            ]
        ).float()
