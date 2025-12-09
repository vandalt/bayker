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
    pos_grid = (
        np.linspace(*ra_bounds, num=npts),
        np.linspace(*dec_bounds, num=npts),
    )
    if not vectorized:
        colin = np.empty((len(pos_grid[1]), len(pos_grid[0])))
        for j, ra in enumerate(pos_grid[0]):
            for i, dec in enumerate(pos_grid[1]):
                kpmod = forward_binary({"cr": cr, "dra": ra, "ddec": dec}, kpo, "radec")
                colin[i, j] = data.dot(kpmod)
    else:
        ra, dec = np.meshgrid(pos_grid[0], pos_grid[1])
        signal = forward_binary({"cr": cr, "dra": ra, "ddec": dec}, kpo, "radec")
        colin = data.dot(signal)
    if return_pos:
        return colin, pos_grid
    else:
        return colin
