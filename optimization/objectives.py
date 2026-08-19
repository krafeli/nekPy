from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from nekPy.postprocessor import PostProcessor

def misfit(Cf, Cfobs, bounds=None, dh=0.05, **kwargs):

    proc  = PostProcessor(Cf,    **kwargs)
    proc0 = PostProcessor(Cfobs, **kwargs)

    nu = proc.get_nu()
    x, y, z = proc.get_coords()

    if bounds is None:
        x1, x2 = 0.5, x.max()
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

    # get the observed field
    nu0 = proc0.get_nu()
    dudy, dwdy = proc0.differentiate(['u', 'w'], ['y'])
    Cf0 = nu * np.sqrt(dudy ** 2 + dwdy ** 2)
    proc0.add_field('Cf', Cf0)
    _, _, _, Cf0i = proc0.box_itp(x1, x2, y1, y1, z1, z2, dh, ['Cf'])

    J = np.mean((Cfi - Cf0i) ** 2) / np.mean(Cf0i ** 2)

    vmin, vmax = np.nanpercentile(Cf0i, (1, 99))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.contourf(X, Z, Cf0i, vmin=vmin, vmax=vmax, levels=256)
    plt.savefig(Path(Cf).parent / 'Cf0.png', dpi=1000)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.contourf(X, Z, Cfi, vmin=vmin, vmax=vmax, levels=256)
    plt.savefig(Path(Cf).parent / 'Cf.png', dpi=1000)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.contourf(X, Z, Cf0i - Cfi, cmap='bwr', levels=256)
    plt.savefig(Path(Cf).parent / 'misfit.png', dpi=1000)
    return float(J)