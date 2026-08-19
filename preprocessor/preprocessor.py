from pathlib import Path
from nekPy.preprocessor.mesh import Mesh
from nekPy.preprocessor.bc import BoundaryCondition
from nekPy.utils.nektools import ParFile, SizeFile, msh2nek
from nekPy.utils.bash import copy, mkdir

class PreProcessor():

    def __init__(self, outdir, usr, par, size, name=None, msh=None, re2=None, ma2=None, additional_files=None):
        
        self.outdir = Path(outdir)
        mkdir(outdir)

        if additional_files is not None:
            for f in additional_files:
                f = Path(f)
                copy(f, self.outdir / f.name)

        usr = Path(usr)
        par = Path(par)
        size= Path(size)

        self.usr_origin = usr
        self.par_origin = par
        self.size_origin = size

        self.name = name if name else usr.stem

        dusr = copy(usr,  (self.outdir / self.name).with_suffix(".usr"))
        dpar = copy(par,  (self.outdir / self.name).with_suffix(".par"))
        dsize = copy(size, (self.outdir / "SIZE").with_suffix(""))

        self.usrfile = dusr
        self.parfile = dpar
        self.sizefile = dsize
    
        self.parameters = ParFile(dpar)
        self.size = SizeFile(dsize)        
        self.msh = Mesh(mshfile=msh) if msh else None
        self.re2 = Path(re2) if re2 else None
        self.ma2 = Path(ma2) if ma2 else None

        self.bc = None
        self.bcstate = 0

    def __str__(self):
        return (
            f"PreProcessor:\n"
            f"  name      = {self.name}\n"
            f"  outdir    = {self.outdir}\n"
            f"  usrfile   = {self.usrfile}\n"
            f"  parfile   = {self.parfile}\n"
            f"  sizefile  = {self.sizefile}\n"
            f"  msh       = {self.msh}\n"
            f"  re2       = {self.re2}\n"
            f"  ma2       = {self.ma2}\n"
            f"  bcstate   = {self.bcstate}\n"
            f"  bc        = {self.bc}"
        )

    def generate_mesh(self, k, eta, Lx, Ly, Lz, Lin=15, N=None, Nin=None, Nx=None, Ny=None, Nz=None, show=False):
        outfile = (self.outdir / f"{self.name}_{k}_{eta}").with_suffix(".msh")
        self.msh = Mesh(outfile=outfile, k=k, eta=eta, Lx=Lx, Ly=Ly, Lz=Lz, Lin=Lin, N=N, Nin=Nin, Nx=Nx, Ny=Ny, Nz=Nz)
        self.msh.generate(show=show)

    def msh2nek(self, **kwargs):
        if self.msh.mshfile is None:
            raise ValueError("No msh file provided")
        msh2nek(self.outdir, self.msh.mshfile.stem, self.name, **kwargs)

    def generate_bc(self, blfile, mode, loc, **kwargs):
        if self.bcstate != 0:
            raise ValueError("BC has already been generated")
        Rek = -self.parameters.get('VELOCITY', 'viscosity')
        self.bc = BoundaryCondition(blfile, mode, loc, Rek, self.outdir, **kwargs)
        self.bc.generate()
        self.bcstate = 1



        
    
    
    
        
