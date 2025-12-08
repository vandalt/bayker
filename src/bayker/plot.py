from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from xara.kpo import KPO


# TODO: Add optional data histogram to the side
def plot_data(
    kpo: KPO, fig: Figure | None = None, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    fig = fig or plt.figure(figsize=(8, 4))
    if ax is None:
        if len(fig.axes) == 0:
            ax = plt.gca()
        else:
            ax = fig.axes[0]
    if kpo.kp.ndim == 1:
        ax.errorbar(kpo.x, kpo.kp, kpo.ekp, mfc="w", fmt="k.", capsize=2)
    else:
        for i in range(kpo.kp.shape[0]):
            ax.errorbar(kpo.x, kpo.kp[i], kpo.ekp[i], mfc="w", fmt="k.", capsize=2)
    ax.set_xlabel("Kernel Phase index")
    ax.set_ylabel("Kernel phase [rad]")
    return fig, ax
