def plot_triangular_example():
    import matplotlib.pyplot as plt
    import numpy as np

    from .. import get_triangular_parameters, triangular_pdf

    J = 5
    alphas = [0.1, 0.1, 0.1, 0.1, 0.1]
    thresholds = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]

    # Represent the thresholds as vertical lines
    for threshold in thresholds:
        plt.axvline(threshold, color="gray", linestyle="--")

    for j in range(J):
        a, b, c = get_triangular_parameters(j, J, alphas, thresholds)
        print(f"Class {j}: a={a}, b={b}, c={c}")
        x = np.linspace(-0.2, 1.2, 500)
        y = [triangular_pdf(xi, a, b, c) for xi in x]
        plt.plot(x, y, label=f"Class {j}")
        plt.fill_between(x, y, step="pre", alpha=0.4)
    plt.xlim(-0.2, 1.2)
    plt.legend()
    plt.savefig("triangular.png")


def plot_asymmetric_triangular_example():
    import matplotlib.pyplot as plt
    import numpy as np

    from .. import get_asymmetric_triangular_parameters, triangular_pdf

    J = 5
    alphas = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
    thresholds = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]

    # Represent the thresholds as vertical lines
    for threshold in thresholds:
        plt.axvline(threshold, color="gray", linestyle="--")

    for j in range(J):
        a, b, c = get_asymmetric_triangular_parameters(j, J, alphas, thresholds)
        print(f"Class {j}: a={a}, b={b}, c={c}")
        x = np.linspace(-0.2, 1.2, 500)
        y = [triangular_pdf(xi, a, b, c) for xi in x]
        plt.plot(x, y, label=f"Class {j}")
        plt.fill_between(x, y, step="pre", alpha=0.4)
    plt.xlim(-0.2, 1.2)
    plt.legend()
    plt.savefig("triangular.png")
