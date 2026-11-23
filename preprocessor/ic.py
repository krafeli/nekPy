import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from pathlib import Path

from nekPy.utils.bash import mkdir
from nekPy.utils.io import read_pkl


class InitialCondition:
 
    def __init__(self, blfile, xloc, Rek, Lin, Lout, outdir=None, fname=None, h=10.0):

        self.blfile = Path(blfile)
        self.bl = read_pkl(blfile)
        self.xloc = xloc
        self.Rek = Rek
        self.Lin = Lin
        self.Lout = Lout
        self.h = h

        self.outdir = None

        if outdir is not None:
            self.outdir = Path(outdir)
            mkdir(self.outdir)


        if fname is None:
            self.fname = "ic.txt"
        else:
            self.fnames = fname
            

        # General quantities
        self.Rei = None
        self.nui = None
        self.Ui = None
        self.sloc = None
        self.kL = None
        self.uk = None
        self.d99L = None
        self.d99k = None
        self.kd99 = None

        # Blade
        self.sin = None
        self.xin = None
        self.ic_file = None


    def __str__(self):

        s = (
            f"BoundaryCondition(\n"
            f"  xloc   = {self.xloc}\n"
            f"  Rek    = {self.Rek}\n"
            f"  Rei    = {self.Rei}\n"
            f"  nui    = {self.nui}\n"
            f"  Ui     = {self.Ui}\n"
            f"  sloc   = {self.sloc}\n"
            f"  kL     = {self.kL}\n"
            f"  uk     = {self.uk}\n"
            f"  d99L   = {self.d99L}\n"
            f"  d99k   = {self.d99k}\n"
            f"  kd99   = {self.kd99}"
        )

        if self.mode == "blade":
            s += (
                f"\n  Lin    = {self.Lin}\n"
                f"  h      = {self.h}\n"
                f"  sin    = {self.sin}\n"
                f"  xin    = {self.xin}\n"
                f"  ic     = {self.ic_file}\n"
            )

        return s + "\n)"

    def generate(self, nx=1000, ny=500, plot_ic=True,):

        bl = self.bl
        loc = self.xloc
        Rek_des = self.Rek

        self.Rei = bl["Re"]
        self.nui = 1.0 / self.Rei
        nu = self.nui

        x = bl["x"]
        s_wall = bl["s"]
        ut = bl["ut"]
        un = bl["un"]
        d_arr = bl["d"]
        Uinf = bl["Uinf"]
        d99 = bl["d99"]
        fct = bl["fct"]

        ut_itp  = interp1d(x, ut, axis=0)
        un_itp  = interp1d(x, un, axis=0)
        d_itp   = interp1d(x, d_arr, axis=0)
        U_itp   = interp1d(x, Uinf)
        s_itp   = interp1d(x, s_wall)
        d99_itp = interp1d(x, d99)
        fct_itp = interp1d(x, fct)
        x_of_s  = interp1d(s_wall, x)

        self.sloc = float(s_itp(loc))
        self.Ui = float(U_itp(loc))
        fct = float(fct_itp(loc))

        ut_c = ut_itp(loc)
        d_c = d_itp(loc)
        d99_c = float(d99_itp(loc))

        dq = np.append(0.0, np.geomspace(1e-6, 5e1, 9999))
        utc_prof_itp = interp1d(d_c, ut_c, bounds_error=False, fill_value=(ut_c[0], ut_c[-1]))
        utq = utc_prof_itp(dq)
        dL = dq / fct
        
        Rek = utq * dL / nu
        Rek_dL = interp1d(dL, Rek, bounds_error=False, fill_value="extrapolate")
        self.kL = root_scalar(lambda kk: Rek_dL(kk) - Rek_des, bracket=[dL.min(), dL.max()]).root
        
        uk_dL = interp1d(dL, utq, bounds_error=False, fill_value=(utq[0], utq[-1]))
        self.uk = float(uk_dL(self.kL))
        self.sin = self.sloc - self.Lin * self.kL
        self.sout = self.sloc + self.Lout * self.kL
        self.xin = float(x_of_s(self.sin))
        self.d99L = d99_c / fct
        self.kd99 = self.kL / self.d99L
        self.d99k = self.d99L / self.kL

        print("Generating IC data...")
        yq_ic = np.append(0.0, np.geomspace(1e-6, self.h+1e-6, ny-1))
        sq_ic = np.linspace(self.sin-1e-6, self.sout+1e-6, nx)

        ## IC
        x, y = [], []
        ut, un = [], []

        for si in sq_ic:
            xi = x_of_s(si)
            fct_i = fct_itp(xi)
            ut_i = ut_itp(xi) / self.uk
            un_i = un_itp(xi) / self.uk
            d_i = d_itp(xi) / fct_i / self.kL      
            ut_i_itp = interp1d(d_i, ut_i,bounds_error=False, fill_value=(ut_i[0], ut_i[-1]))
            un_i_itp = interp1d(d_i, un_i,bounds_error=False, fill_value=(un_i[0], un_i[-1]))
            x.append((si - self.sloc) / self.kL)         
            y.append(yq_ic)
            ut.append(ut_i_itp(yq_ic))          
            un.append(un_i_itp(yq_ic))

        x, y = np.array(x), np.array(y)
        ut, un = np.array(ut), np.array(un)

        X = np.repeat(x[:, None], len(yq_ic), axis=1)
        Y = np.tile(yq_ic, (len(sq_ic), 1))

        if self.outdir:
            self.ic_file = self.outdir / self.fname
            np.savetxt(self.ic_file, np.column_stack((X.ravel(), Y.ravel(), ut.ravel(), un.ravel())),
            fmt="%.12e", header=f"x/k y/k ut/uk un/uk (Rek={Rek_des:.0f}, xc={self.xloc:.2f})")
        print(f"Wrote {nx*ny:.0f} points to {self.ic_file}")

        if plot_ic:
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            cf1 = axs[0].contourf(X, Y, ut, levels=512)
            cbar1 = fig.colorbar(cf1, ax=axs[0])
            cbar1.set_label(r"$u_t/u_k$")
            axs[0].set_ylabel(r"$y/k$")
            axs[0].set_title(r"$u_t/u_k$")
            
            axs[0].contour(X, Y, ut, levels=[1.], colors='k')

            
            cf2 = axs[1].contourf(X, Y, un, levels=512)
            cbar2 = fig.colorbar(cf2, ax=axs[1])
            cbar2.set_label(r"$u_n/u_k$")
            axs[1].set_xlabel(r"$x/k$")
            axs[1].set_ylabel(r"$y/k$")
            axs[1].set_title(r"$u_n/u_k$")

            for ax in axs:
                ax.axhline(1, xmin=X.min(), xmax=X.max(), ls='--', lw=0.5, c='silver')
                ax.set_xlim(X.min(), X.max())
                ax.set_ylim(Y.min(), Y.max())
            fig.suptitle(f"Initial condition ($Re_k={Rek_des:.0f}$, $x_c={self.xloc:.2f}$)")
            fig.tight_layout()
            if self.outdir:
                plt.savefig(self.outdir / "ic.png", dpi=800)
            plt.close()
            
        return self
