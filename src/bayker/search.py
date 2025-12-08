import numpy as np
from xara.kpo import KPO

from bayker.model import forward_binary


def get_colinearity_map(
    kpo: KPO,
    npts: int,
    ra_bounds: tuple,
    dec_bounds: tuple | None = None,
    cr: float = 0.1,
    return_pos: bool = False,
):
    data = np.array(kpo.KPDT).squeeze()
    if data.ndim > 1:
        raise ValueError("Colinearity maps implemented for a single frame only")
    if dec_bounds is None:
        dec_bounds = ra_bounds
    pos_grid = {
        "ra": np.linspace(*ra_bounds, num=npts),
        "dec": np.linspace(*dec_bounds, num=npts),
    }
    colin = np.empty((len(pos_grid["dec"]), len(pos_grid["ra"])))
    for j, ra in enumerate(pos_grid["ra"]):
        for i, dec in enumerate(pos_grid["dec"]):
            kpmod = forward_binary({"cr": cr, "dra": ra, "ddec": dec}, kpo, "radec")
            colin[i, j] = data.dot(kpmod)
    if return_pos:
        return colin, pos_grid
    else:
        return colin
