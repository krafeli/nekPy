from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from nekPy.postprocessor import PostProcessor

def misfit(Cfpth, Cfobspth, obsnu, bounds=None, dh=0.05, **kwargs):

    proc  = PostProcessor(Cfpth,    **kwargs)
    proc0 = PostProcessor(Cfobspth, **kwargs)

    nu = proc.get_nu()
    x, y, z = proc.get_coords()

    if bounds is None:
        x1, x2 = 0.5, x.max() * 0.8
        y1 = y.min()
        z1, z2 = z.min(), z.max()
    else:
        x1, x2 = bounds[0], bounds[1]
        y1 = y.min()
        z1, z2 = bounds[2], bounds[3]

    dudy, dwdy = proc.differentiate(['u', 'w'], ['y'])
    Cf = nu * np.sqrt(dudy ** 2 + dwdy ** 2)
    proc.add_field('Cf', Cf)
    X, Y, Z, Cfi = proc.box_itp(x1, x2, y1, y1, z1, z2, dh, ['Cf'])
    # substract far field
    Xl, Yl, Zl, Cfil = proc.box_itp(x1, x2, y1, y1, z.max(), z.max(), dh, ['Cf'])
    Cfi_hat = Cfi - Cfil[:, None]

    # get the observed field
    nu0 = obsnu
    dudy, dwdy = proc0.differentiate(['u', 'w'], ['y'])
    Cf0 = nu0 * np.sqrt(dudy ** 2 + dwdy ** 2)
    proc0.add_field('Cf', Cf0)
    _, _, _, Cf0i = proc0.box_itp(x1, x2, y1, y1, z1, z2, dh, ['Cf'])
    _, _, _, Cf0il = proc0.box_itp(x1, x2, y1, y1, z.max(), z.max(), dh, ['Cf'])
    Cf0i_hat = Cf0i - Cf0il[:, None]
    J = np.mean((Cfi_hat - Cf0i_hat) ** 2) / np.mean(Cf0i ** 2)

    vmin, vmax = np.nanpercentile(Cf0i, (1, 99))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.contourf(X, Z, Cf0i_hat, vmin=vmin, vmax=vmax, levels=256)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(Path(Cfpth).parent / 'Cf0.png', dpi=1000)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.contourf(X, Z, Cfi_hat, vmin=vmin, vmax=vmax, levels=256)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(Path(Cfpth).parent / 'Cf.png', dpi=1000)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.contourf(X, Z, (Cfi_hat - Cf0i_hat)**2 / np.mean(Cf0i_hat ** 2), cmap='bwr', levels=256)
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(Path(Cfpth).parent / 'misfit.png', dpi=1000)
    return float(J)