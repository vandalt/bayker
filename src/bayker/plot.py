from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from bayker.model import KernelModel


# TODO: Support multi-integration kpfits
def plot_kernel(
    model: KernelModel,
    parameters: np.ndarray | dict[str, float | np.ndarray] | None = None,
    n_samples: int = 100,
    residuals: bool = True,
    include_sigma: bool = False,
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

    fig, axs = plt.subplots(
        1 + residuals, 1, figsize=(12, 4 + 4 * residuals), sharex=True
    )

    if residuals:
        axd, axr = axs
    else:
        axs = [axs]
        axd = axs[0]
    axd.errorbar(
        model.kpo.x,
        model.kpo.kp,
        yerr=model.kpo.ekp,
        fmt="k.",
        capsize=2,
        mfc="w",
        label="Data",
    )
    if include_sigma:
        axd.errorbar(
            model.kpo.x,
            model.kpo.kp,
            yerr=inflated_ekp,
            ecolor="r",
            fmt="none",
            alpha=0.5,
            capsize=2,
            label="Inflated errors",
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
                axr.errorbar(
                    model.kpo.x,
                    res,
                    yerr=model.kpo.ekp,
                    fmt="k.",
                    capsize=2,
                    mfc="w",
                )
                if include_sigma:
                    axr.errorbar(
                        model.kpo.x,
                        res,
                        yerr=inflated_ekp,
                        ecolor="r",
                        fmt="none",
                        alpha=0.5,
                        capsize=2,
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
                    axr.errorbar(
                        model.kpo.x,
                        resi,
                        yerr=model.kpo.ekp,
                        alpha=0.1,
                        fmt="k.",
                        capsize=2,
                        mfc="w",
                        label="Data",
                    )
                    if include_sigma:
                        axr.errorbar(
                            model.kpo.x,
                            resi,
                            yerr=inflated_ekp,
                            ecolor="r",
                            fmt="none",
                            alpha=0.005,
                            capsize=2,
                        )
        else:
            raise ValueError(
                f"Unexpected dimension {ndim} for parameters. Should be 1 or 2."
            )

    axd.legend()

    return fig, axs
