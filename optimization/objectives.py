from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from nekPy.postprocessor import PostProcessor

def misfit(Cfpth, Cfobspth, nuobs, bounds, dh=0.05, **kwargs):

    proc  = PostProcessor(Cfpth,    **kwargs)
    proc0 = PostProcessor(Cfobspth, **kwargs)

    nu = proc.get_nu()
    x, y, z = proc.get_coords()
    dudy, dwdy = proc.differentiate(['u', 'w'], ['y'])
    Cf = 2*nu*np.sqrt(dudy ** 2 + dwdy ** 2)
    proc.add_field('Cf', Cf)

    nu0 = nuobs
    x0, y0, z0 = proc0.get_coords()
    dudy0, dwdy0 = proc0.differentiate(['u', 'w'], ['y'])
    Cf0 = 2*nu0*np.sqrt(dudy0 ** 2 + dwdy0 ** 2)
    proc0.add_field('Cf', Cf0)

    x1, x2 = bounds[0], bounds[1]
    z1, z2 = bounds[2], bounds[3]

    # interpolate fields and far-field
    X, Y, Z, Cfi = proc.box_itp(x1, x2, 0., 0., z1, z2, dh, ['Cf'])
    Xl, Yl, Zl, Cfil = proc.box_itp(x1, x2, 0., 0., z.max(), z.max(), dh, ['Cf'])
    Cfi_hat = Cfi - Cfil[:, None]

    # observed field
    _, _, _, Cf0i = proc0.box_itp(x1, x2, 0., 0., z1, z2, dh, ['Cf'])
    _, _, _, Cf0il = proc0.box_itp(x1, x2, 0., 0., z0.max(), z0.max(), dh, ['Cf'])
    Cf0i_hat = Cf0i - Cf0il[:, None]

    J = np.mean((Cfi_hat - Cf0i_hat) ** 2) / np.mean(Cf0i_hat ** 2)

    vmin, vmax = np.nanpercentile(Cf0i, (1, 99))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.contourf(X, Z, Cf0i_hat, vmin=vmin, vmax=vmax, levels=256)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.colorbar(im, ax=ax)
    plt.savefig(Path(Cfpth).parent / 'Cf0.png', dpi=1000)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.contourf(X, Z, Cfi_hat, vmin=vmin, vmax=vmax, levels=256)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.colorbar(im, ax=ax)
    plt.savefig(Path(Cfpth).parent / 'Cf.png', dpi=1000)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    im = ax.contourf(X, Z, (Cfi_hat - Cf0i_hat)**2 / np.mean(Cf0i_hat ** 2), cmap='bwr', levels=256)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.colorbar(im, ax=ax)
    plt.savefig(Path(Cfpth).parent / 'misfit.png', dpi=1000)
    plt.close()
    return float(J)