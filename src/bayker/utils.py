import numpy as np
from xara.kpo import KPO


def seppa2radec(
    sep: float | np.ndarray, pa: float | np.ndarray
) -> tuple[float | np.ndarray]:
    """Convert separation and position angle (PA) to RA and Dec

    PA = 0 is along the Y axis (north) and RA increases to the left (east)

    :param sep: Separation in mas
    :param pa: Position angle in deg
    :return: RA and Dec in mas
    """
    pa_rad = np.deg2rad(pa)

    # PA = 0 is along the Y axis (North)
    ddec = sep * np.cos(pa_rad)
    # RA increases to the left (East)
    dra = sep * np.sin(pa_rad)

    return dra, ddec


def radec2seppa(
    ra: float | np.ndarray, dec: float | np.ndarray
) -> tuple[float | np.ndarray]:
    """Convert RA and Dec to separation and position angle (PA)

    PA = 0 is along the Y axis (north) and RA increases to the left (east)

    :param ra: RA in mas
    :param dec: Dec in mas
    :return: Separation and PA in mas and deg, respectively
    """
    sep = np.sqrt(ra**2 + dec**2)
    # RA is like y, dec like x
    pa = np.rad2deg(np.arctan2(ra, dec))

    return sep, pa


# TODO: Upstream into xara?
def average_kpo(kpo: KPO) -> KPO:
    """Average kernel phase observations

    For ``kpo.KPDT`` with ``nsets`` datasets with shape ``(nints, nkp)``,
    each dataset is averaged along the ``nints`` axis.

    If ``kpo.KPSIG`` is already set, performs a weighted average using inverse variance
    weighting (weights = 1/sigma^2) and propagates errors accordingly.
    Otherwise, computes a simple median and stores the standard error in ``kpo.KPSIG``.

    :param kpo: Kernel phase observations object
    :return: Returns a copy of the kernel phase observations with all integrations averaged
    """
    kpo = kpo.copy()
    nsets = len(kpo.KPDT)
    if all([d.shape[0] == 1 for d in kpo.KPDT]):
        return kpo

    has_errors = len(kpo.KPSIG) > 0 and all([sig is not None for sig in kpo.KPSIG])

    if not has_errors:
        kpo.KPSIG = [None] * nsets

    for i in range(nsets):
        nints = kpo.KPDT[i].shape[0]

        if has_errors:
            # Weighted average with inverse variance weighting
            weights = 1.0 / (kpo.KPSIG[i] ** 2)
            sum_weights = np.sum(weights, axis=0, keepdims=True)
            avg_val = np.sum(kpo.KPDT[i] * weights, axis=0, keepdims=True) / sum_weights
            # Error propagation for weighted average
            avg_err = np.sqrt(1.0 / np.sum(weights, axis=0))
            kpo.KPSIG[i] = avg_err
        else:
            # Simple median with standard error
            avg_val = np.expand_dims(np.median(kpo.KPDT[i], axis=0), 0)
            avg_err = np.sqrt(np.var(kpo.KPDT[i], axis=0) / (nints - 1))
            kpo.KPSIG[i] = avg_err

        kpo.KPDT[i] = avg_val
    return kpo


def calibrate_kpo(kpo_sci: KPO, kpo_cal: KPO) -> KPO:
    """Calibrate a science KPO with a reference KPO

    :param kpo_sci: Science KPO object
    :param kpo_cal: Reference (calibrator) KPO object
    :return: Returns a copy of ``kpo_sci`` with the calibrated ``KPDT`` and error-propagated ``KPSIG``
    """
    kpo = kpo_sci.copy()
    kpo.KPDT = list(np.array(kpo_sci.KPDT) - np.array(kpo_cal.KPDT))
    kpo.KPSIG = list(
        np.sqrt(np.array(kpo_sci.KPSIG) ** 2 + np.array(kpo_cal.KPSIG) ** 2)
    )
    return kpo
