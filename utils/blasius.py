import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

#blasius ode
def solve_blasius(etas):

    def _ode(eta, y):
        f, fp, fpp = y
        return [fp, fpp, -0.5 * f * fpp]

    def _ivp(fpp0_guess, etas):
        y0 = [0, 0, fpp0_guess]
        sol = solve_ivp(_ode, (etas.min(), etas.max()), y0, t_eval=etas, rtol=1e-12, atol=1e-12)
        return sol

    def _obj(fpp0, args):
        etas = args
        sol = _ivp(fpp0, etas)
        return sol.y[1, -1] - 1

    result = root_scalar(_obj, bracket=[0, 10], method='brentq', args=(etas,), rtol=1e-12, xtol=1e-12)
    fpp0 = result.root

    sol = _ivp(fpp0, etas)

    eta = sol.t
    f = sol.y[0]
    fp = sol.y[1]
    fpp = sol.y[2]

    return eta, f, fp, fpp
