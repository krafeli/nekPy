import logging
import numpy as np
from pathlib import Path

from .mpi import comm, rank, nranks, rank0

from nekPy.postprocessor.pysemtools.io.ppymech.neksuite import pynekread, pynekwrite
from nekPy.postprocessor.pysemtools.datatypes.msh import Mesh
from nekPy.postprocessor.pysemtools.datatypes.coef import Coef
from nekPy.postprocessor.pysemtools.datatypes.field import FieldRegistry
from nekPy.postprocessor.pysemtools.datatypes.utils import extrude_2d_sem_mesh
from nekPy.postprocessor.pysemtools.datatypes.msh_partitioning import MeshPartitioner
from nekPy.postprocessor.pysemtools.datatypes.msh_connectivity import MeshConnectivity
from nekPy.postprocessor.pysemtools.interpolation.probes import Probes

from nekPy.utils.io import write_json, write_pkl
from nekPy.utils.nektools import ParFile

class PostProcessor():

    def __init__(self, path, dtype=np.double, msh=None, fld=None, coef=None, create_connectivity=False, get_coef=False, verbose=True):

        if not verbose:
            logging.disable(logging.CRITICAL)

        self.path = str(path)
        self.dtype = np.dtype(dtype)
        self.comm = comm
        self.rank = rank
        self.nranks = nranks
        self.rank0 = rank0
        self.verbose = verbose and self.rank0

        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"Unsupported dtype {self.dtype}.")

        self.msh = msh if msh is not None else Mesh(self.comm, create_connectivity=create_connectivity)
        self.fld = fld if fld is not None else FieldRegistry(self.comm)
        self.msh_con = None

        pynekread(self.path,self.comm, msh=self.msh if msh is None else None, fld=self.fld, overwrite_fld=True, data_dtype=self.dtype)

        self.coef = coef if coef is not None else (Coef(self.msh, self.comm, get_area=True) if get_coef else None)

        self.dim = 2 if np.all(self.msh.z == 0.0) else 3
        self.shape = self.msh.x.shape
        self.params = {}
        self.scalar_mapping = {}

        if self.verbose:
            print("Available fields:", list(self.fld.registry.keys()))
            print("Detected dim:", self.dim)

    def read_par(self, parfile=None):
        if parfile is None:
            parfile = list(Path(self.path).parent.glob('*.par'))
            if len(parfile) != 1:
                raise IOError("No or more than one .par files found.")
            parfile = parfile[0]

        self.parfile = ParFile(parfile)
        return self.parfile

    def get_nu(self, parfile=None):
        params = self.read_par(parfile=parfile)
        return 1./-params.get('VELOCITY', 'viscosity')

    def get_Re(self, parfile=None):
        params = self.read_par(parfile=parfile)
        return -params.get('VELOCITY', 'viscosity')

    def apply_mapping(self, mapping):
        if self.verbose:
            print("Renaming fields according to according to mapping...")
        for k, v in mapping.items():
            if self.verbose:
                print(f"{k} -> {v}")
            self.fld.rename_registry_key(old_key=k, new_key=v)

    def save(self, out_path):
        out_path = Path(out_path)
        if self.verbose:
            print(f"Writing outfile: {out_path}")
        pynekwrite(str(out_path), self.comm, msh=self.msh, fld=self.fld, write_mesh=True)
        write_json(self.scalar_mapping, out_path.parent / f"scalar_mapping.json")

    def available_fields(self, verbose=True):
        if verbose: print("Available fields:", list(self.fld.registry.keys()))
        return list(self.fld.registry.keys())

    def get_time(self):
        return self.fld.t

    def get_coords(self):
        return self.msh.x, self.msh.y, self.msh.z

    def has_field(self, name):
        return name in self.fld.registry

    def get_field(self, name):
        if not self.has_field(name):
            raise KeyError(f"Field '{name}' not found. Available: {list(self.fld.registry.keys())}")
        return self.fld.registry[name]

    def get_fields(self, names):
        outfields = []
        for name in names:
            outfields.append(self.get_field(name))
        return outfields

    def add_field(self, name, arr, verbose=True):
        if arr.shape != self.shape:
            raise ValueError("Shape mismatch between added field and mesh.")

        self.fld.add_field(self.comm, field_name=name, field=arr, dtype=self.dtype)
        scalar_pos = self.fld.registry_pos[name]
        scalar_name = scalar_pos.split('_')[0][0] + scalar_pos.split('_')[1]
        self.scalar_mapping[scalar_name] = name
        if self.verbose and verbose:
            print(f"Added field '{name}' to registry position {scalar_name}")

    def add_fields(self, names, arrs, verbose=True):
        for name, arr in zip(names, arrs):
            self.add_field(name, arr, verbose=verbose)

    def remove_field(self, name, verbose=True):
        if not self.has_field(name):
            raise KeyError(f"Field '{name}' not found. Available: {list(self.fld.registry.keys())}")
        keys_to_remove = [k for k, v in self.scalar_mapping.items() if v == name]
        for k in keys_to_remove:
            del self.scalar_mapping[k]
        del self.fld.registry[name]
        if hasattr(self.fld, "registry_pos") and name in self.fld.registry_pos:
            del self.fld.registry_pos[name]
        if self.verbose and verbose:
            print(f"Removed field '{name}'")

    def remove_fields(self, names, verbose=True):
        for name in names:
            self.remove_field(name, verbose=verbose)

    def clean(self, verbose=True):
        self.remove_fields(list(self.fld.registry.keys()), verbose=verbose)

    def get_dict(self):
        x, y, z = self.get_coords()
        out = {
            'x': x,
            'y': y,
            'z': z,
        }
        for k, v in self.fld.registry.items():
            out[k] = v
        return out

    def save_pkl(self, out_path):
        out_path = str(out_path)
        if not out_path.endswith('.pkl'):
            out_path = out_path + '.pkl'
        if self.verbose:
            print(f"Writing outfile: {out_path}")
        write_pkl(self.get_dict(), out_path)

    def get_box_fields(self, bounds, fields):
        out_fields = []
        (x1, x2), (y1, y2), (z1, z2) = bounds
        x, y, z = self.get_coords()
        m = (x>=x1) & (x<=x2) & (y>=y1) & (y<=y2) & (z>=z1) & (z<=z2)
        for field in fields:
            fi = self.get_field(field)
            out_fields.append(fi[m])

        return np.column_stack((x[m], y[m], z[m])), out_fields


    # field smoothing (NOT WORKING)
    def smooth(self, fld, inplace=False, iters=1):
        msh_con = MeshConnectivity(self.comm, self.msh, rel_tol=1e-5)
        target = self.get_field(fld)
        fld = target.copy()
        for _ in range(iters):
            fld = msh_con.dssum(field=fld, msh=self.msh, average="multiplicity")
        if inplace:
            target[...] = fld
            return target
        return fld

    # derivatives
    def differentiate(self, fields, directions, smooth=0, verbose=False):

        name_fmt = "d{f}d{d}"
        if not self.coef:
            self.coef = Coef(self.msh, self.comm, get_area=True)

        if self.dim == 2 and ("z" in directions or "w" in fields):
            raise ValueError("Mismatch of directions/fields and dimensions.")

        outfields = []
        for field in fields:

            if not self.has_field(field):
                raise KeyError(f"Cannot differentiate '{field}': not in registry.")

            fieldarray = self.get_field(field)

            for direction in directions:

                name = name_fmt.format(f=field, d=direction)

                if self.has_field(name):
                    deriv = self.get_field(name)
                    if smooth: deriv = self.smooth(name, inplace=True, iters=smooth)

                    outfields.append(deriv)
                    continue

                if self.verbose or verbose:
                    print(f"Calculating the derivative {name}...")

                if direction == "x":
                    deriv = self.coef.dudxyz(fieldarray,
                        self.coef.drdx,
                        self.coef.dsdx,
                        self.coef.dtdx if self.dim == 3 else None
                    )
                elif direction == "y":
                    deriv = self.coef.dudxyz(
                        fieldarray,
                        self.coef.drdy,
                        self.coef.dsdy,
                        self.coef.dtdy if self.dim == 3 else None
                    )
                elif direction == "z":
                    if self.dim != 3:
                        raise ValueError("Requested z-derivative in 2D case.")
                    deriv = self.coef.dudxyz(
                        fieldarray,
                        self.coef.drdz,
                        self.coef.dsdz,
                        self.coef.dtdz
                    )
                else:
                    raise ValueError(f"Unknown direction '{direction}'. Must be one of ('x','y','z').")


                self.add_field(name, deriv, verbose=verbose)
                if smooth != 0: deriv = self.smooth(name, inplace=True, iters=smooth)
                outfields.append(deriv)

        if self.verbose:
            print("Done calculating derivatives.")
            print("Available fields:", list(self.fld.registry.keys()))

        return outfields

    def compute_grads(self, smooth=0):
        if self.dim == 3:
            fields = ["u", "v", "w", "p", "t"]
            directions = ["x", "y", "z"]
        elif self.dim == 2:
            fields = ["u", "v", "p", "t"]
            directions = ["x", "y"]
        else:
            raise NotImplementedError(f"Dimension {self.dim} not implemented.")
        self.differentiate(fields, directions, smooth=smooth)

    def integrate(self, fields, mask=None, average=False, operation=None):

        if mask is None:
            mask = np.ones_like(self.msh.x, dtype=int)
        elif callable(mask):
            mask = np.asarray(mask(*self.get_coords()), dtype=bool)
        elif isinstance(mask, np.ndarray):
            mask = np.asarray(mask, dtype=int)
        else:
           raise ValueError("mask must be array or callable.")

        if self.coef is None:
            self.coef = Coef(self.msh, self.comm, get_area=True)

        out_fields = []

        for field in fields:
            fi = self.get_field(field)
            if callable(operation):
                fi = operation(fi)
            B = self.coef.B * mask
            I = self.coef.glsum(fi * B, self.comm, dtype=self.dtype)
            if average: I = I / self.coef.glsum(B, self.comm, dtype=self.dtype)
            out_fields.append(I)

        return out_fields

    # interpolation
    def interpolate(self, xyz, fields, mask=None, bounds=None, fill_value=None,
                    max_pts=128, shape=-1, probes=None, return_probes=False):

        for f in fields:
            if not self.has_field(f):
                raise ValueError(f"Field {f} not available. Available: {list(self.fld.registry.keys())}")


        msh_itp = self.msh
        fld_itp = self.fld

        if self.dim == 2:
            msh_itp, fld_itp = extrude_2d_sem_mesh(self.comm, lz=self.msh.lx, msh=self.msh, fld=self.fld)

        if bounds is not None:
            x, y, z = msh_itp.x, msh_itp.y, msh_itp.z
            if callable(bounds):
                con = bounds(x, y, z)
            else:
                raise ValueError("Bounds must be a callable f(x, y, [z]).")

            mp = MeshPartitioner(self.comm, msh=msh_itp, conditions=[con])
            alg = "load_balanced_linear"
            msh_itp = mp.create_partitioned_mesh(msh_itp, partitioning_algorithm=alg)
            fld_itp = mp.create_partitioned_field(fld_itp, partitioning_algorithm=alg)

        if self.rank0:
            xyz = np.asarray(xyz, dtype=self.dtype)
            n, m = xyz.shape
            if m != self.dim:
                raise ValueError("xyz dimension mismatch")
            if m == 2:
                xyz = np.column_stack((xyz, np.zeros((n, 1), dtype=self.dtype)))
            if mask is None:
                mask_arr = np.ones(n, dtype=bool)
            elif callable(mask):
                mask_arr = np.asarray(mask(xyz), dtype=bool)
            else:
                mask_arr = np.asarray(mask, dtype=bool)
                if mask_arr.shape != (n,):
                    raise ValueError("mask must be length n")
            pts = xyz[mask_arr]
        else:
            pts = None
            mask_arr = None
            n = None

        if probes is None:
            probes = Probes(self.comm, probes=pts, msh=msh_itp, max_pts=max_pts,
                            point_interpolator_type="multiple_point_legendre_numpy",
                            find_points_comm_pattern="point_to_point",
                            write_coords=False)

        field_list = [fld_itp.registry[fi] for fi in fields]
        probes.interpolate_from_field_list(0, field_list, self.comm, write_data=False)

        if self.rank0:
            data = probes.interpolated_fields[:, 1:]
            err = probes.itp.err_code
            err_mask = np.full(len(err), False, dtype=bool)
            err_mask[(err == 0)] = True
            if fill_value is not None:
                data[err_mask, :] = fill_value

            out_fields = []
            for dat in data.T:
                full_data = np.full(n, fill_value, dtype=self.dtype)
                full_data[mask_arr] = dat
                out_fields.append(full_data.reshape(shape))
        else:
            out_fields = None
        out_fields = self.comm.bcast(out_fields, root=0)

        if return_probes: return out_fields, probes
        return out_fields

    def box_itp(self, xmn, xmx, ymn, ymx, zmn, zmx, h, fields):
        
        def _make_axis(mn, mx, h):
            if np.isclose(mn, mx):
                return np.array([mn], dtype=self.dtype)
            return np.arange(mn, mx + h, h, dtype=self.dtype)

        h = np.asarray(h, dtype=self.dtype)

        if h.ndim == 0:
            hx = hy = hz = h.item()
        elif h.shape == (3,):
            hx, hy, hz = h
        else:
            raise ValueError("h must be a scalar or an array-like of length 3: [hx, hy, hz].")

        xq = _make_axis(xmn, xmx, hx)
        yq = _make_axis(ymn, ymx, hy)
        zq = _make_axis(zmn, zmx, hz)
        
        Xq, Yq, Zq = np.meshgrid(xq, yq, zq, indexing='ij')
        XYZq = np.column_stack([Xq.ravel(), Yq.ravel(), Zq.ravel()])
        itp_fields = self.interpolate(XYZq, fields, shape=Xq.shape)
        
        slc = tuple(0 if len(q) == 1 else slice(None) for q in (xq, yq, zq))
        Xq = Xq[slc]
        Yq = Yq[slc]
        Zq = Zq[slc]
        itp_fields = [f[slc] for f in itp_fields]
        
        return Xq, Yq, Zq, *itp_fields
        
    
    def var(self, var_path=None, sqrt=False):
        pth = Path(self.path)
        
        if var_path is None:
            var_path = pth.parent / str(pth.name).replace('avg', 'rms')
        
        var_proc = PostProcessor(var_path,  dtype=self.dtype, msh=self.msh, create_connectivity=False, verbose=self.verbose)
        var_fields = {}
        for name in self.available_fields(verbose=False):
            f, fvar = self.get_field(name), var_proc.get_field(name)
            var_name = 2*name
            var_field = np.maximum(fvar - f**2, 0.0)
            if sqrt: var_field = np.sqrt(var_field)
            self.add_field(var_name, var_field)
            var_fields[var_name] = var_field
        del var_proc
        return var_fields
    
    def cov(self, cov_path=None):
        pth = Path(self.path)

        if cov_path is None:
            cov_path = pth.parent / str(pth.name).replace("avg", "rm2")

        cov_proc = PostProcessor(cov_path, dtype=self.dtype, msh=self.msh, create_connectivity=False, verbose=self.verbose)

        cov_fields = {}

        # rm2 field 'u' contains mean(u*v)
        uv = cov_proc.get_field("u") - self.get_field("u") * self.get_field("v")
        self.add_field("uv", uv)
        cov_fields["uv"] = uv

        if self.dim > 2:
            # rm2 field 'v' contains mean(v*w)
            vw = cov_proc.get_field("v") - self.get_field("v") * self.get_field("w")
            self.add_field("vw", vw)
            cov_fields["vw"] = vw

            # rm2 field 'w' contains mean(u*w)
            uw = cov_proc.get_field("w") - self.get_field("u") * self.get_field("w")
            self.add_field("uw", uw)
            cov_fields["uw"] = uw

        del cov_proc
        return cov_fields

    def stats(self, var_path=None, cov_path=None):
        out = {}
        out.update(self.var(var_path=var_path))
        out.update(self.cov(cov_path=cov_path))
        return out
    
    def reynolds_stress(self, var_path=None, cov_path=None, remove_fields=False):
        """
        Return Reynolds stress tensor R_ij = <u_i' u_j'>.
        """
        
        var_fields = ["uu", "vv"] if self.dim == 2 else ["uu", "vv", "ww"]
        cov_fields = ["uv"] if self.dim == 2 else ["uv", "uw", "vw"]

        if not all(self.has_field(f) for f in var_fields):
            self.var(var_path=var_path)

        if not all(self.has_field(f) for f in cov_fields):
            self.cov(cov_path=cov_path)

        uu = self.get_field("uu")
        vv = self.get_field("vv")
        uv = self.get_field("uv")

        if self.dim == 2:
            z = np.zeros_like(uu)
            return np.array([
                [uu, uv, z],
                [uv, vv, z],
                [z,  z,  z],
            ])

        ww = self.get_field("ww")
        uw = self.get_field("uw")
        vw = self.get_field("vw")
        if remove_fields:
            self.remove_fields(var_fields + cov_fields)
            if self.has_field('pp'): self.remove_field('pp')
            if self.has_field('tt'): self.remove_field('tt')

        R = np.array([[uu, uv, uw],[uv, vv, vw],[uw, vw, ww]])
        return R
    
    def ensure_derivative(self, field, direction, smooth=False):
        name = f"d{field}d{direction}"
        if not self.has_field(name):
            if self.verbose:
                print(f"Field '{name}' not available. Computing '{name}'..")
            self.differentiate([field], [direction], smooth=smooth, verbose=self.verbose)

    def grad_u(self, smooth=False, remove_fields=False):
        """
        Compute gradU = ∇U and return as (3,3,npts,lx1,lx1,lx1).
        """
        dudx = self.differentiate(['u'], ['x'], smooth=smooth)[0]
        dudy = self.differentiate(['u'], ['y'], smooth=smooth)[0]
        dvdx = self.differentiate(['v'], ['x'], smooth=smooth)[0]
        dvdy = self.differentiate(['v'], ['y'], smooth=smooth)[0]
        names = ['dudx', 'dudy', 'dvdx', 'dvdy']
        if self.dim == 3:
            dudz = self.differentiate(['u'], ['z'], smooth=smooth)[0]
            dvdz = self.differentiate(['v'], ['z'], smooth=smooth)[0]
            dwdx = self.differentiate(['w'], ['x'], smooth=smooth)[0]
            dwdy = self.differentiate(['w'], ['y'], smooth=smooth)[0]
            dwdz = self.differentiate(['w'], ['z'], smooth=smooth)[0]
            names += ['dudz', 'dvdz', 'dwdx', 'dwdy', 'dwdz']
        else:
            dudz = np.zeros_like(dudx)
            dvdz = np.zeros_like(dudx)
            dwdx = np.zeros_like(dudx)
            dwdy = np.zeros_like(dudx)
            dwdz = np.zeros_like(dudx)

        if remove_fields: self.remove_fields(names)

        G = np.array([
            [dudx, dudy, dudz],
            [dvdx, dvdy, dvdz],
            [dwdx, dwdy, dwdz],
        ])  # (3,3,npts,lx1,lx1,lx1)

        return G

    def compute_strain_tensor(self, add_to_field=False, smooth=False, out_name="S"):

        # shape: (3,3,npts,lx1,lx1,lx1)
        G = self.grad_u(smooth=smooth)
        S =  0.5*(G + np.transpose(G, axes=(1, 0, 2, 3, 4, 5)))

        if add_to_field:
            self.add_field(f"{out_name}xx", S[0, 0, :])
            self.add_field(f"{out_name}xy", S[0, 1, :])
            self.add_field(f"{out_name}xz", S[0, 2, :])
            self.add_field(f"{out_name}yx", S[1, 0, :])
            self.add_field(f"{out_name}yy", S[1, 1, :])
            self.add_field(f"{out_name}yz", S[1, 2, :])
            self.add_field(f"{out_name}zx", S[2, 0, :])
            self.add_field(f"{out_name}zy", S[2, 1, :])
            self.add_field(f"{out_name}zz", S[2, 2, :])
        return S

    def compute_stress_tensor(self, visc=1., add_to_field=False, smooth=False, out_name="tau"):
        """
        Compute tau = visc * (gradU + gradU^T).
        returns dict with components:
          tau_xx, tau_xy, tau_xz,
          tau_yx, tau_yy, tau_yz,
          tau_zx, tau_zy, tau_zz
        """
        # shape: (3,3,npts,lx1,lx1,lx1)
        tau = 2*visc*(self.compute_strain_tensor(smooth=smooth))
        if add_to_field:
            self.add_field(f"{out_name}xx", tau[0, 0, :])
            self.add_field(f"{out_name}xy", tau[0, 1, :])
            self.add_field(f"{out_name}xz", tau[0, 2, :])
            self.add_field(f"{out_name}yy", tau[1, 1, :])
            self.add_field(f"{out_name}yz", tau[1, 2, :])
            self.add_field(f"{out_name}zz", tau[2, 2, :])
        return tau

    def lambda2(self, smooth=0, out_name="l2"):
        G = self.grad_u(smooth=smooth, remove_fields=True)
        S = 0.5 * (G + np.swapaxes(G, 0, 1))
        O = 0.5 * (G - np.swapaxes(G, 0, 1))
        SS = np.einsum('ik...,kj...->ij...', S, S)
        OO = np.einsum('ik...,kj...->ij...', O, O)
        M = SS + OO

        M_ = np.moveaxis(M.reshape(3, 3, -1), -1, 0)  # (npts, 3, 3)
        eigs = np.linalg.eigvalsh(M_)
        lam2 = eigs[:, 1].reshape(M.shape[2:])

        self.add_field(f"{out_name}", lam2)

        return lam2

    def ekin(self, smooth=False, out_name="E"):
        u, v, w = self.get_fields(['u', 'v', 'w'])
        E = 0.5 * np.sqrt(u**2 + v**2 + w**2)
        self.add_field(f"{out_name}", E)
        return E
