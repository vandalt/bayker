from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from bayker.model import KernelModel


def plot_data(
    x: np.ndarray, kp: np.ndarray, ekp: np.ndarray, fig=None, ax=None, inflated_ekp=None
):
    fig = fig or plt.gcf()
    if ax is None:
        if len(fig.axes) == 0:
            ax = plt.gca()
        else:
            ax = fig.axes[0]
    data_kwargs = dict(
        mfc="w",
        fmt="k.",
        capsize=2,
    )
    inf_kwargs = dict(
        ecolor="r",
        fmt="none",
        alpha=0.5,
        capsize=2,
    )
    if kp.ndim == 1:
        ax.errorbar(x, kp, ekp, label="Data", **data_kwargs)
        if inflated_ekp is not None:
            ax.errorbar(
                x,
                kp,
                yerr=inflated_ekp,
                label="Inflated errors",
                **inf_kwargs,
            )
    else:
        for i in range(kp.shape[0]):
            ax.errorbar(
                x,
                kp[i],
                ekp[i],
                label="Data" if i == 0 else None,
                **data_kwargs,
            )
            if inflated_ekp is not None:
                ax.errorbar(
                    x,
                    kp[i],
                    yerr=inflated_ekp[i],
                    **inf_kwargs,
                    label="Inflated errors" if i == 0 else None,
                )


# TODO: Support multi-integration kpfits
def plot_kernel(
    model: KernelModel,
    parameters: np.ndarray | dict[str, float | np.ndarray] | None = None,
    n_samples: int = 100,
    residuals: bool = True,
    include_sigma: bool = False,
    fig: Figure = None,
    axs: Axes = None,
) -> tuple[Figure, Axes]:
    """Plot the Kernel phase data and model

    :param model: Model whose data and basis will be used
    :param parameter: Parameter values. If an array, must be compatible with ``KernelModel.forward`` or have shape ``(nparam, nsamples_total)``
                      If a dict, will be combined with fixed parameters, but if fixed parameters are specified here they have priority. Can map to parameter values or to arrays of parameter samples.
    :param n_samples: Number of samples to draw if samples are passed as parameters. Ignored for 1D parameters.
    :param residuals: Whether to show a residuals panel. Defaults to `True`.
    :param include_sigma: Whether to show a residuals panel. Whether to show the inflated error bars when a `sigma` parameter is in the model. Defaults to `False`.
    """
    show_model = parameters is not None
    if show_model:
        if not isinstance(parameters, dict):
            parameters = dict(zip(model.keys(), parameters))
        ndim = np.array(list(parameters.values())[0]).ndim + 1
        if ndim == 1:
            parameters_single = parameters.copy()
        elif ndim == 2:
            parameters_single = {k: np.median(v) for k, v in parameters.items()}
        else:
            raise ValueError(
                f"Unexpected dimension {ndim} for parameters. Should be 1 or 2."
            )

    residuals = residuals and show_model
    include_sigma = include_sigma and "sigma" in parameters
    if include_sigma:
        inflated_ekp = np.sqrt(model.kpo.ekp**2 + parameters_single["sigma"] ** 2)
    else:
        inflated_ekp = None

    if fig is None or axs is None:
        fig, axs = plt.subplots(
            1 + residuals, 1, figsize=(12, 4 + 4 * residuals), sharex=True
        )

    if residuals:
        axd, axr = axs
    else:
        axs = [axs]
        axd = axs[0]
    plot_data(
        model.kpo.x,
        model.kpo.kp,
        model.kpo.ekp,
        fig=fig,
        ax=axd,
        inflated_ekp=inflated_ekp,
    )
    axd.set_ylabel("Kernel Phase [rad]")

    axs[-1].set_xlabel("Kernel Phase Index")
    axd.legend()

    if show_model:
        if residuals:
            axr.axhline(0.0, linestyle="--", color="C0")
            axr.set_ylabel("Residuals [rad]")

        if ndim == 1:
            mod_kp = model.forward(parameters)
            axd.plot(
                model.kpo.x,
                mod_kp,
                label="Model",
                color="C0",
            )
            if residuals:
                res = model.kpo.kp - mod_kp
                plot_data(
                    model.kpo.x,
                    res,
                    model.kpo.ekp,
                    inflated_ekp=inflated_ekp,
                    fig=fig,
                    ax=axr,
                )
        elif ndim == 2:
            posterior_preds = model.get_posterior_pred(parameters, n_samples)
            for i in range(n_samples):
                axd.plot(
                    model.kpo.x,
                    posterior_preds[i],
                    color="C0",
                    alpha=0.1,
                    label="Model samples" if i == 0 else None,
                )
                if residuals:
                    resi = model.kpo.kp - posterior_preds[i]
                    plot_data(
                        model.kpo.x,
                        resi,
                        model.kpo.ekp,
                        inflated_ekp=inflated_ekp,
                        fig=fig,
                        ax=axr,
                    )
        else:
            raise ValueError(
                f"Unexpected dimension {ndim} for parameters. Should be 1 or 2."
            )

    axd.legend()

    return fig, axs
