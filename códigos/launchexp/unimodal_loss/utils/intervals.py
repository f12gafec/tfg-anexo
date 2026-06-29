import numpy as np


def get_intervals(n):
    """Get n evenly-spaced intervals in [0,1].

    Parameters
    ----------
    n : int
            Number of intervals.

    Returns
    -------
    intervals: list
            List of intervals.
    """

    points = np.linspace(1e-9, 1 - 1e-9, n + 1)
    intervals = []
    for i in range(0, points.size - 1):
        intervals.append((points[i], points[i + 1]))

    return intervals
