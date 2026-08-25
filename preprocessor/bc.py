import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, minimize_scalar

from pathlib import Path

from nekPy.utils.bash import mkdir
from nekPy.utils.io import read_pkl, write_json
from nekPy.utils.blasius import solve_blasius


class BoundaryCondition:

    def __init__(self, blfile, mode, xloc, Rek, outdir=None, Lin=None, fnames=None, h=10.0, fit_range_k=(0.0, 10.0)):

        self.blfile = Path(blfile)
        self.bl = read_pkl(blfile)
        self.mode = mode.lower()
        self.xloc = xloc
        self.Rek = Rek
        self.Lin = Lin
        self.h = h
        self.fit_range_k = fit_range_k

        self.outdir = None

        if outdir is not None:
            self.outdir = Path(outdir)
            mkdir(self.outdir)

        if self.mode not in ["blade", "blasius"]: raise ValueError("Mode must be 'blade' or 'blasius'")

        if fnames is None:
            if self.mode == "blade": self.fnames = ["inlet.txt", "top.txt"]
            elif self.mode == "blasius": self.fnames = ["blas.txt"]
        else:
            self.fnames = fnames
        if self.mode == "blade" and self.Lin is None: raise ValueError("Mode 'blade' requires inflow location Lin")
        if self.mode == "blasius" and self.Lin: raise ValueError("Mode 'blasius' doesnt use inflow Lin")

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
        self.inflow_file = None
        self.top_file = None

        # Blasius
        self.blas_file = None
        self.Ue = None
        self.blas_file = None
        self.xloc_shifted_L = None
        self.xloc_shifted_k = None
        self.ukb_shifted_raw = None
        self.mse = None
        self.d99L_b = None
        self.d99k_b = None
        self.kd99_b = None

    def __str__(self):

        s = (
            f"BoundaryCondition(\n"
            f"  mode   = {self.mode}\n"
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
                f"  inflow = {self.inflow_file}\n"
                f"  top    = {self.top_file}"
            )

        elif self.mode == "blasius":
            s += (
                f"\n  Ue             = {self.Ue}\n"
                f"  xloc_shifted_L = {self.xloc_shifted_L}\n"
                f"  xloc_shifted_k = {self.xloc_shifted_k}\n"
                f"  ukb_shifted    = {self.ukb_shifted_raw}\n"
                f"  mse            = {self.mse}\n"
                f"  d99L_b         = {self.d99L_b}\n"
                f"  d99k_b         = {self.d99k_b}\n"
                f"  kd99_b         = {self.kd99_b}\n"
                f"  blas           = {self.blas_file}"
            )

        return s + "\n)"

    def generate(self, save_config=True, verbose=True):

        bl = self.bl
        loc = self.xloc
        Rek_des = self.Rek

        self.Rei = bl["Re"]
        self.nui = 1.0 / self.Rei
        nu = self.nui

        # Blade boundary conditions
        if self.mode == "blade":

            if len(self.fnames) != 2: raise ValueError("Mode 'blade' requires two filenames: [inflow_filename, top_filename]")

            x = bl["x"]
            s_wall = bl["s"]
            ut = bl["ut"]
            un = bl["un"]
            d_arr = bl["d"]
            Uinf = bl["Uinf"]
            d99 = bl["d99"]
            fct = bl["fct"]

            ut_itp = interp1d(x, ut, axis=0)
            un_itp = interp1d(x, un, axis=0)
            d_itp = interp1d(x, d_arr, axis=0)
            U_itp = interp1d(x, Uinf)
            s_itp = interp1d(x, s_wall)
            d99_itp = interp1d(x, d99)
            fct_itp = interp1d(x, fct)
            x_of_s = interp1d(s_wall, x)

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
            Rek_check = self.uk * self.kL / nu

            self.sin = self.sloc - self.Lin * self.kL
            self.xin = float(x_of_s(self.sin))
            x_in_loc = (self.sin - self.sloc) / self.kL

            if verbose:
                print("---------------------------------INPUTS------------------------------------")
                print(f"Re_inf   = {self.Rei:.0e}")
                print(f"xc       = {loc:.6f}")
                print(f"sc       = {self.sloc:.6f}")
                print(f"Uinf_xc  = {self.Ui:.6f}")
                print(f"Rek_des  = {Rek_des:.0f}")

                print("--------------------------BLADE BOUNDARY LAYER------------------------------")
                print(f"k        = {self.kL:.6e}L")
                print(f"u_k      = {self.uk:.6f}Uinf")
                print(f"************** Sanity check: achieved Rek = {Rek_check:.1f}")
                print(f"s_in     = {self.sin:.6f}L")
                print(f"x_in     = {self.xin:.6f}L")
                print(f"x_in_loc = {x_in_loc:.1f}k")

            self.d99L = d99_c / fct
            self.kd99 = self.kL / self.d99L
            self.d99k = self.d99L / self.kL

            print("Writing inflow and top data...")
            fct_in = float(fct_itp(self.xin))
            ut_in = ut_itp(self.xin) / self.uk
            un_in = un_itp(self.xin) / self.uk
            d_in = d_itp(self.xin) / fct_in / self.kL
            yq_in = np.append(0.0, np.geomspace(1e-6, 50.0, 99999))
            ut_in_itp = interp1d(d_in, ut_in, bounds_error=False, fill_value=(ut_in[0], ut_in[-1]))
            un_in_itp = interp1d(d_in, un_in, bounds_error=False, fill_value=(un_in[0], un_in[-1]))

            if self.outdir:
                self.inflow_file = self.outdir / self.fnames[0]
                np.savetxt(self.inflow_file, np.column_stack((yq_in, ut_in_itp(yq_in), un_in_itp(yq_in))),
                           header=f"y/k   ut/uk   un/uk (Rek={Rek_des:.0f}, xloc={loc:.2f})", fmt="%.12e")

            sq = np.linspace(self.sin-1e-6, s_wall.max(), 100000)
            x_top, ut_top, un_top = [], [], []

            for si in sq:
                xi = float(x_of_s(si))
                fct_i = float(fct_itp(xi))
                ut_i = ut_itp(xi) / self.uk
                un_i = un_itp(xi) / self.uk
                d_i = d_itp(xi) / fct_i / self.kL
                ut_i_itp = interp1d(d_i, ut_i, bounds_error=False, fill_value=(ut_i[0], ut_i[-1]))
                un_i_itp = interp1d(d_i, un_i, bounds_error=False, fill_value=(un_i[0], un_i[-1]))
                x_top.append((si - self.sloc) / self.kL)
                ut_top.append(ut_i_itp(self.h))
                un_top.append(un_i_itp(self.h))

            if self.outdir:
                self.top_file = self.outdir / self.fnames[1]
                np.savetxt(self.top_file, np.column_stack((x_top, ut_top, un_top)),
                           fmt="%.12e", header=f"xloc/k   ut(y={self.h:.1f}k)/uk   un(y={self.h:.1f}k)/uk   (Rek={Rek_des:.0f})")

            print(f"Wrote {self.inflow_file}")
            print(f"Wrote {self.top_file}")

        # Blasius boundary condition
        elif self.mode == "blasius":

            if len(self.fnames) != 1: raise ValueError("Mode 'blasius' requires one filename: [blasius_filename]")

            eta_blas = np.linspace(0.0, np.sqrt(10.0), 100000) ** 2
            eta_blas, f, fp, fpp = solve_blasius(eta_blas)
            ub_blas = fp
            vb_blas = 0.5 * (eta_blas * fp - f)

            if self.outdir:
                self.blas_file = self.outdir / self.fnames[0]
                np.savetxt(self.blas_file, np.column_stack((eta_blas, ub_blas, vb_blas)), fmt="%.12e")

            etab_raw, _, ub, _ = solve_blasius(np.linspace(0.0, 50.0, 10000))

            x = bl["x"]
            ut_itp = interp1d(x, bl["ut"], axis=0)
            d_itp = interp1d(x, bl["d"], axis=0)
            U_itp = interp1d(x, bl["Uinf"])
            s_itp = interp1d(x, bl["s"])
            d99_itp = interp1d(x, bl["d99"])
            fct_itp = interp1d(x, bl["fct"])

            self.sloc = float(s_itp(loc))
            self.Ui = float(U_itp(loc))
            fct = float(fct_itp(loc))

            u = ut_itp(loc)
            eta = d_itp(loc) / fct
            self.d99L = float(d99_itp(loc)) / fct
            self.Ue = float(np.interp(self.d99L, eta, u))

            Rek = eta * u / nu
            ki = np.where(Rek >= Rek_des)[0][0]

            self.kL = np.interp(Rek_des, Rek[ki - 1:ki + 1], eta[ki - 1:ki + 1])
            self.uk = np.interp(self.kL, eta, u)
            self.d99k = self.d99L / self.kL
            self.kd99 = self.kL / self.d99L

            etab = etab_raw / fct
            ukb_fixed_x = np.interp(self.kL, etab, ub)

            def blasius_mse(x_shifted):
                fct_shifted = np.sqrt(self.Ue / (nu * x_shifted))
                etab_shifted = etab_raw / fct_shifted
                eta_fit_min = self.fit_range_k[0] * self.kL
                eta_fit_max = self.fit_range_k[1] * self.kL
                fit_mask = (eta >= eta_fit_min) & (eta <= eta_fit_max)
                eta_fit = eta[fit_mask]
                u_fit = u[fit_mask] / self.uk
                ub_fit = np.interp(eta_fit, etab_shifted, ub)
                ukb_fit = np.interp(self.kL, etab_shifted, ub)
                diff2 = (ub_fit / ukb_fit - u_fit) ** 2
                return np.trapezoid(diff2, eta_fit) / (eta_fit[-1] - eta_fit[0])

            opt = minimize_scalar(blasius_mse, bounds=(0.0, 0.5), method="bounded", tol=1e-12)

            self.xloc_shifted_L = opt.x
            self.xloc_shifted_k = self.xloc_shifted_L / self.kL
            self.mse = opt.fun
            fct_shifted = np.sqrt(self.Ue / (nu * self.xloc_shifted_L))

            etab_shifted = etab_raw / fct_shifted

            self.ukb_shifted_raw = np.interp(self.kL, etab_shifted, ub)

            self.d99L_b = np.interp(0.99, ub, etab_shifted)
            self.d99k_b = self.d99L_b / self.kL
            self.kd99_b = self.kL / self.d99L_b


            L_fmt = ".8e"
            k_fmt = ".8e"
            label_width = 42

            print(f"\nLocation x/L = {loc:.5f}")
            print("-" * 72)
            print(f"{'Nearest available x':<{label_width}} = {self.xloc:{L_fmt}}L")
            print(f"{'Arc chord length s':<{label_width}} = {self.sloc:{L_fmt}}L")
            print(f"{'Ue = u(eta_99)':<{label_width}} = {self.Ue:.8e}")
            print(f"{'nu':<{label_width}} = {self.nui:.8e}")
            print()
            print(f"{'Target roughness Reynolds number':<{label_width}} = {Rek_des:.0f}")
            print(f"{f'k for Re_k = {Rek_des:.0f}':<{label_width}} = {self.kL:{L_fmt}}L")
            print(f"{'u_k blade boundary layer':<{label_width}} = {self.uk:.8e}")
            print()
            print("Blasius, fixed x_w:")
            print()
            print("Blasius, matched MSE:")
            print(f"\033[31m{'Shifted x':<{label_width}} = {self.xloc_shifted_L:{L_fmt}}L = {self.xloc_shifted_k:{k_fmt}}k")
            print(f"{'u_k,Blasius,matched':<{label_width}} = {self.ukb_shifted_raw:.8e}\033[0m")
            print(f"{'Profile MSE':<{label_width}} = {self.mse:.8e}")
            print()
            print(f"{'Blasius eta_99':<{label_width}} = {self.d99L_b:{L_fmt}}L = {self.d99k_b:{k_fmt}}k")
            print(f"{'Blasius k/eta_99':<{label_width}} = {self.kd99_b:.8e}")
            print(f"\nWrote {self.blas_file}")

        if save_config:
            config = {
                "mode": self.mode,
                "blfile": str(self.blfile),
                "xloc": self.xloc,
                "Rek": self.Rek,
                "Rei": self.Rei,
                "nui": self.nui,
                "Ui": self.Ui,
                "sloc": self.sloc,
                "kL": self.kL,
                "uk": self.uk,
                "d99L": self.d99L,
                "d99k": self.d99k,
                "kd99": self.kd99,
            }

            if self.mode == "blade":
                config.update({
                    "Lin": self.Lin,
                    "h": self.h,
                    "sin": self.sin,
                    "xin": self.xin,
                    "inflow_file": str(self.inflow_file),
                    "top_file": str(self.top_file),
                })

            elif self.mode == "blasius":
                config.update({
                    "Ue": self.Ue,
                    "fit_range_k": self.fit_range_k,
                    "xloc_shifted_L": self.xloc_shifted_L,
                    "xloc_shifted_k": self.xloc_shifted_k,
                    "ukb_shifted_raw": self.ukb_shifted_raw,
                    "mse": self.mse,
                    "d99L_b": self.d99L_b,
                    "d99k_b": self.d99k_b,
                    "kd99_b": self.kd99_b,
                    "blas_file": str(self.blas_file),
                })
            if self.outdir:
                self.config_file = self.outdir / "bc.json"
                write_json(config, str(self.config_file))

        return self