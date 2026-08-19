from pathlib import Path
from nekPy.preprocessor.mesh import Mesh
from nekPy.preprocessor.bc import BoundaryCondition
from nekPy.utils.nektools import ParFile, SizeFile, msh2nek
from nekPy.utils.bash import copy, mkdir


class PreProcessor():

    def __init__(self, outdir, usr, par, size, name=None, msh=None, re2=None, ma2=None, additional_files=None, save_originals=False):

        self.outdir = Path(outdir)
        mkdir(outdir)

        if additional_files is not None:
            for f in additional_files:
                f = Path(f)
                copy(f, self.outdir / f.name)

        usr = Path(usr)
        par = Path(par)
        size = Path(size)

        self.usr_origin = usr
        self.par_origin = par
        self.size_origin = size

        self.name = name if name else usr.stem

        dusr = copy(usr, (self.outdir / self.name).with_suffix(".usr"))
        dpar = copy(par, (self.outdir / self.name).with_suffix(".par"))
        dsize = copy(size, (self.outdir / "SIZE").with_suffix(""))

        self.usrfile = dusr
        self.parfile = dpar
        self.sizefile = dsize

        self.parameters = ParFile(dpar)
        self.size = SizeFile(dsize)

        self.msh_origin = Path(msh) if msh else None
        self.re2_origin = Path(re2) if re2 else None
        self.ma2_origin = Path(ma2) if ma2 else None

        if msh:
            dmsh = copy(msh, (self.outdir / self.name).with_suffix(".msh"))
            self.msh = Mesh(mshfile=dmsh)
        else:
            self.msh = None

        if re2:
            dre2 = copy(re2, (self.outdir / self.name).with_suffix(".re2"))
            self.re2 = dre2
        else:
            self.re2 = None

        if ma2:
            dma2 = copy(ma2, (self.outdir / self.name).with_suffix(".ma2"))
            self.ma2 = dma2
        else:
            self.ma2 = None

        if save_originals:
            original_dir = self.outdir / "original_config"
            mkdir(original_dir)
            copy(self.usr_origin, original_dir / self.usr_origin.name)
            copy(self.par_origin, original_dir / self.par_origin.name)
            copy(self.size_origin, original_dir / self.size_origin.name)
            if self.msh_origin:
                copy(self.msh_origin, original_dir / self.msh_origin.name)
            if self.re2_origin:
                copy(self.re2_origin, original_dir / self.re2_origin.name)
            if self.ma2_origin:
                copy(self.ma2_origin, original_dir / self.ma2_origin.name)


        self.bc = None
        self.bcstate = 0

    def __str__(self):
        return (
            f"PreProcessor:\n"
            f"  name        = {self.name}\n"
            f"  outdir      = {self.outdir}\n"
            f"\n"
            f"  usr_origin  = {self.usr_origin}\n"
            f"  par_origin  = {self.par_origin}\n"
            f"  size_origin = {self.size_origin}\n"
            f"  msh_origin  = {self.msh_origin}\n"
            f"  re2_origin  = {self.re2_origin}\n"
            f"  ma2_origin  = {self.ma2_origin}\n"
            f"\n"
            f"  usrfile     = {self.usrfile}\n"
            f"  parfile     = {self.parfile}\n"
            f"  sizefile    = {self.sizefile}\n"
            f"  msh         = {self.msh}\n"
            f"  re2         = {self.re2}\n"
            f"  ma2         = {self.ma2}\n"
            f"\n"
            f"  bcstate     = {self.bcstate}\n"
            f"  bc          = {self.bc}"
        )

    def generate_mesh(self, k, eta, Lx, Ly, Lz, Lin=15, N=None, Nin=None, Nx=None, Ny=None, Nz=None, show=False):
        outfile = (self.outdir / f"{self.name}_{k}_{eta}").with_suffix(".msh")
        self.msh = Mesh(outfile=outfile, k=k, eta=eta, Lx=Lx, Ly=Ly, Lz=Lz, Lin=Lin, N=N, Nin=Nin, Nx=Nx, Ny=Ny, Nz=Nz)
        self.msh.generate(show=show)

    def msh2nek(self, **kwargs):
        if self.msh is None:
            raise ValueError("No mesh provided")
        msh2nek(self.outdir, self.msh.mshfile.stem, self.name, **kwargs)
        self.re2 = (self.outdir / self.name).with_suffix(".re2")
        self.ma2 = (self.outdir / self.name).with_suffix(".ma2")

    def generate_bc(self, blfile, mode, loc, **kwargs):
        if self.bcstate != 0:
            raise ValueError("BC has already been generated")
        Rek = -self.parameters.get('VELOCITY', 'viscosity')
        self.bc = BoundaryCondition(blfile, mode, loc, Rek, self.outdir, **kwargs)
        self.bc.generate()
        self.bcstate = 1
