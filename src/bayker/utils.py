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

def average_kpo(kpo: KPO) -> KPO:
    kpo = kpo.copy()
