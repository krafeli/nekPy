import gmsh, math, json
import numpy as np

from pathlib import Path
from nekPy.utils.io import write_json


def smallest_element(L, N, r):
    if abs(r - 1) < 1e-12:
        return L / N
    return L * (r - 1) / (r**N - 1)


def largest_element(L, N, r):
    if abs(r - 1) < 1e-12:
        return L / N
    return smallest_element(L, N, r) * r**(N - 1)


def progression(N, r):
    if r == 1: r = r+1e-10
    dx1 = (r - 1) / (r**N - 1)
    x = np.cumsum([dx1 * r**i for i in range(N)])
    return np.array(x)


def progression_ratio(L, N, dx1, tol=1e-12):
    lo, hi = 0.0, 10.0
    while hi - lo > tol:
        r = (lo + hi) / 2
        s = dx1 * (r**N - 1)/(r - 1)
        if s > L:
            hi = r
        else:
            lo = r
    return (lo + hi) / 2


def reverse_progression(heights):
    layer_sizes = np.diff(np.concatenate(([0.0], heights)))
    reversed_heights = np.cumsum(layer_sizes[::-1])
    reversed_heights[-1] = 1.0
    return reversed_heights


class Mesh:

    def __init__(self, mshfile=None, outfile=None, k=None, eta=None, Lx=None, Ly=None, Lz=None, Lin=15,
        N=None, Nin=None, Nx=None, Ny=None, Nz=None, Nh=8):

        if mshfile is None and outfile is None:
            raise ValueError("Provide either mshfile or outfile")
        if mshfile is not None:
            self.mshfile = Path(mshfile)
            self.outfile = self.mshfile.with_suffix("")
            if not self.mshfile.exists(): 
                raise FileNotFoundError(f"Mesh file does not exist: {self.mshfile}")
        else:
            self.outfile = Path(outfile)
            self.mshfile = self.outfile

        self.k = k
        self.eta = eta
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.Lin = Lin

        self.N = N
        self.Nin = Nin
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nh = Nh

        self.Nelem = None

        configfile = self.mshfile.with_suffix(".json")
        # load mesh metadata if available
        if mshfile is not None and configfile.exists():
            config = json.loads(configfile.read_text())
            if self.k is None: self.k = config.get("k")
            if self.eta is None: self.eta = config.get("eta")
            if self.Lx is None: self.Lx = config.get("Lx")
            if self.Ly is None: self.Ly = config.get("Ly")
            if self.Lz is None: self.Lz = config.get("Lz")
            if self.N is None: self.N = config.get("N")
            if self.Nin is None: self.Nin = config.get("Nin")
            if self.Nx is None: self.Nx = config.get("Nx")
            if self.Ny is None: self.Ny = config.get("Ny")
            if self.Nz is None: self.Nz = config.get("Nz")
            if self.Nh is None: self.Nh = config.get("Nh")
            if self.Nelem is None: self.Nelem = config.get("Nelem")

    def __str__(self):
        return (
            f"Mesh(\n"
            f"  file  = {self.mshfile},\n"
            f"  k     = {self.k},\n"
            f"  eta   = {self.eta},\n"
            f"  Lx    = {self.Lx},\n"
            f"  Ly    = {self.Ly},\n"
            f"  Lz    = {self.Lz},\n"
            f"  Lin   = {self.Lin},\n"
            f"  N     = {self.N},\n"
            f"  Nin   = {self.Nin},\n"
            f"  Nx    = {self.Nx},\n"
            f"  Ny    = {self.Ny},\n"
            f"  Nz    = {self.Nz},\n"
            f"  Nh    = {self.Nh},\n"
            f"  Nelem = {self.Nelem}\n"
            f")"
        )

    def generate(self, show=False, save_config=True):

        if self.k is None: raise ValueError("k is required for mesh generation")
        if self.eta is None: raise ValueError("eta is required for mesh generation")
        if self.Lx is None: raise ValueError("Lx is required for mesh generation")
        if self.Ly is None: raise ValueError("Ly is required for mesh generation")
        if self.Lz is None: raise ValueError("Lz is required for mesh generation")

        outfile = str(self.outfile)
        k, eta = self.k, self.eta
        Lx, Ly, Lz, Lin = self.Lx, self.Ly, self.Lz, self.Lin
        N, Nin, Nx, Ny, Nz = self.N, self.Nin, self.Nx, self.Ny, self.Nz
        gmsh.initialize()

        occ = gmsh.model.occ
        mod = gmsh.model
        point, line, carc = occ.addPoint, occ.addLine, occ.addCircleArc
        cloop, psurf = occ.addCurveLoop, occ.addPlaneSurface
        pi, sin, cos, sqrt = math.pi, math.sin, math.cos, math.sqrt
        msh = gmsh.model.mesh

        # dimensions
        d = k * eta                             # cylinder diameter
        r = d/2.                                # cylinder radius
        Lx, Ly, Lz = Lx + Lin, Ly, Lz

        # would not touch
        ra = 2. * r - ((1-eta)/d)**2           # outer circle radius
        d = ra * 1.5                            # box edge half-length

        if N is None: N = 5+eta//2              # number of nodes of quarter circle
        if Nin is None: Nin = 2*(N+1) - eta     # inflow grid points
        if Nx is None: Nx = Nin*(Lx-d)//Lin * 1.75           # outflow grid points
        if Ny is None: Ny = 1.0*Ly                            # number of layers for k <= y <= Ly
        if Nz is None: Nz = Nin*(Lz-d)//Lin * 1.5            # grid points in top/bottom boxes

        Nr = N - 1                              # radial circle points
        Nh = self.Nh                            # number of layers for 0 <= y <= k
        Nri = N // 2                            # ogrid radial circle points

        # Progressions
        Pr = 1.25                               # radial circle progression
        Pw = 1.2                                # progression at wall until y=k

        dh = largest_element(abs(k), Nh, Pw)         # largest element for y <= k
        Py = progression_ratio(Ly - abs(k), Ny, dh)  # ensure first layer is dh high

        # progression so that the last element of inflow has the same
        # size as the next element of the box
        ds = (d-ra) / (N/2 - 1)
        Pin = progression_ratio(Lin - d, Nin-1, ds)
        Pout = progression_ratio(Lx - Lin - d, Nx-1, ds)
        Pz = progression_ratio(Lz/2-d, Nz - 1, ds)    # upper/lower progression

        # make all int
        N = int(N) | 1
        Nin = int(Nin)
        Nx = int(Nx)
        Nz = int(Nz)
        Nr = int(Nr)
        Nh = int(Nh)
        Ny = int(Ny)
        Nri = int(Nri)

        ##################################################################################
        ### cylinder circle

        # midpoint
        cm = point(0,0,0)

        # inner circle
        ci1 = point(r, 0, 0)
        ci2 = point(-r, 0, 0)
        cimt = point(0, 0, -r)
        cimb = point(0, 0, r)
        ci_top_left = carc(ci2, point(-r/sqrt(2), 0, -r/sqrt(2)), cimt, center=False)
        ci_top_right = carc(cimt, point(r/sqrt(2), 0, -r/sqrt(2)), ci1, center=False)
        ci_bot_right = carc(ci1, point(r/sqrt(2), 0, r/sqrt(2)), cimb, center=False)
        ci_bot_left = carc(cimb, point(-r/sqrt(2), 0, r/sqrt(2)), ci2, center=False)

        # outer circle
        ca1 = point(ra, 0, 0)
        ca2 = point(-ra, 0, 0)
        camt = point(0, 0, -ra)
        camb = point(0, 0, ra)
        ca_top_left = carc(ca2, point(-ra/sqrt(2), 0, -ra/sqrt(2)), camt, center=False)
        ca_top_right = carc(camt, point(ra/sqrt(2), 0, -ra/sqrt(2)), ca1, center=False)
        ca_bot_right = carc(ca1, point(ra/sqrt(2), 0, ra/sqrt(2)), camb, center=False)
        ca_bot_left = carc(camb, point(-ra/sqrt(2), 0, ra/sqrt(2)), ca2, center=False)

        # connecting lines
        circ_con_l = line(ci2, ca2)
        circ_con_t = line(cimt, camt)
        circ_con_r = line(ci1, ca1)
        circ_con_b = line(cimb, camb)

        # surfaces
        S_circ_tl = cloop([ci_top_left, circ_con_t, ca_top_left, circ_con_l])
        S_circ_tl = psurf([S_circ_tl])
        S_circ_tr = cloop([ci_top_right, circ_con_t, ca_top_right, circ_con_r])
        S_circ_tr = psurf([S_circ_tr])
        S_circ_br = cloop([ci_bot_right, circ_con_r, ca_bot_right, circ_con_b])
        S_circ_br = psurf([S_circ_br])
        S_circ_bl = cloop([ci_bot_left, circ_con_l, ca_bot_left, circ_con_b])
        S_circ_bl = psurf([S_circ_bl])

        # close the quadrants
        qtl = point(-d, 0, -d)
        qtm = point(0, 0, -d)
        qtr = point(d, 0, -d)
        qmr = point(d, 0, 0)
        qbr = point(d, 0, d)
        qbm = point(0, 0, d)
        qbl = point(-d, 0, d)
        qml = point(-d, 0, 0)

        # lines outline
        top_left = line(qtl, qtm)
        top_right = line(qtm, qtr)
        right_top = line(qtr, qmr)
        right_bot = line(qmr, qbr)
        bot_right = line(qbr, qbm)
        bot_left = line(qbm, qbl)
        left_bot = line(qbl, qml)
        left_top = line(qml, qtl)

        # line connections
        top_c = line(qtm, camt)
        right_c = line(qmr, ca1)
        bot_c = line(qbm, camb)
        left_c = line(ca2, qml)

        # surfaces
        S_TL = cloop([top_c, top_left, left_top, left_c, ca_top_left])
        S_TL = psurf([S_TL])
        S_TR = cloop([top_c, top_right, right_top, right_c, ca_top_right])
        S_TR = psurf([S_TR])
        S_BR = cloop([right_c, right_bot, bot_right, bot_c, ca_bot_right])
        S_BR = psurf([S_BR])
        S_BL = cloop([bot_c, bot_left, left_bot, left_c, ca_bot_left])
        S_BL = psurf([S_BL])

        # circle at y=k and O grid
        # midpoint
        ckm = point(0, k, 0)

        # circle points and arcs
        cktr = point(r*cos(-pi/4), k, r*sin(-pi/4))
        cktl = point(r*cos(-3*pi/4), k, r*sin(-3*pi/4))
        ckbr = point(r*cos(pi/4), k, r*sin(pi/4))
        ckbl = point(r*cos(3*pi/4), k, r*sin(3*pi/4))
        ck_top = carc(cktl, ckm, cktr)
        ck_bot = carc(ckbr, ckm, ckbl)
        ck_left = carc(ckbl, ckm, cktl)
        ck_right = carc(cktr, ckm, ckbr)

        # O-grid
        lam = 0.75                                      # smaller = more curved o grid
        rb = 0.75 * r                                  # box radius

        tcp = point(0, k, -lam*r)                  # top center point
        bcp = point(0, k, lam*r)                   # bot center point
        lcp = point(-lam*r, k, 0)                  # left center point
        rcp = point(lam*r, k, 0)                   # right center point
        otr = point(rb*cos(-pi/4), k, rb*sin(-pi/4))
        otl = point(rb*cos(-3*pi/4), k, rb*sin(-3*pi/4))
        obr = point(rb*cos(pi/4), k, rb*sin(pi/4))
        obl = point(rb*cos(3*pi/4), k, rb*sin(3*pi/4))
        otc = carc(otl, bcp, otr)
        obc = carc(obl, tcp, obr)
        olc = carc(obl, rcp, otl)
        orc = carc(obr, lcp, otr)

        # arc-circle connection
        tla = line(cktl, otl)
        tra = line(cktr, otr)
        bra = line(ckbr, obr)
        bla = line(ckbl, obl)

        # surfaces
        S_O = cloop([otc, orc, obc, olc])
        S_O = psurf([S_O])
        S_O1 = cloop([tla, ck_top, tra, otc])
        S_O1 = psurf([S_O1])
        S_O2 = cloop([tra, ck_right, bra, orc])
        S_O2 = psurf([S_O2])
        S_O3 = cloop([bra, ck_bot, bla, obc])
        S_O3 = psurf([S_O3])
        S_O4 = cloop([bla, ck_left, tla, olc])
        S_O4 = psurf([S_O4])

        def add_rect(x1,x2,z1,z2, nx, nz, px=1, pz=1):
            A = point(x1, 0, z1)
            B = point(x1, 0, z2)
            C = point(x2, 0, z1)
            D = point(x2, 0, z2)
            left = line(A, B)
            right = line(C, D)
            top = line(A, C)
            bottom = line(B, D)
            R = cloop([left, top, right, bottom])
            R = psurf([R])
            occ.synchronize()
            msh.setTransfiniteCurve(top, nx, coef=px)
            msh.setTransfiniteCurve(bottom, nx, coef=px)
            msh.setTransfiniteCurve(left, nz, coef=pz)
            msh.setTransfiniteCurve(right, nz, coef=pz)
            msh.setTransfiniteSurface(R, cornerTags=[A, B, C, D])
            return R

        Rt = add_rect(-d, d, -d, -Lz / 2, nx=2 * N - 1, nz=Nz, pz=Pz)
        Rb = add_rect(-d, d, d, Lz / 2, nx=2 * N - 1, nz=Nz, pz=Pz)
        Rint = add_rect(-Lin, -d, -Lz / 2, -d, nx=Nin, nz=Nz, px=-Pin, pz=-Pz)
        Rm = add_rect(-Lin, -d, -d, d, nx=Nin, nz=2 * N - 1, px=-Pin)
        Rinb = add_rect(-Lin, -d, d, Lz / 2, nx=Nin, nz=Nz, px=-Pin, pz=Pz)
        Routt = add_rect(d, Lx - Lin, -Lz / 2, -d, nx=Nx, nz=Nz, px=Pout, pz=-Pz)
        Routm = add_rect(d, Lx - Lin, -d, d, nx=Nx, nz=2 * N - 1, px=Pout)
        Routb = add_rect(d, Lx - Lin, d, Lz / 2, nx=Nx, nz=Nz, px=Pout, pz=Pz)

        # mesh extrusion
        tol = 1e-3
        h0 = progression(Nh, Pw)
        h = progression(Ny, Py)

        if k > 0:
            occ.extrude(occ.getEntitiesInBoundingBox(-Lin - tol, -tol, -Lz / 2 - tol, Lx - Lin + tol, tol, Lz / 2 + tol, dim=2),
                        0, abs(k), 0, numElements=[1] * Nh, heights=h0, recombine=True)
            occ.synchronize()
            occ.extrude(occ.getEntitiesInBoundingBox(-1e5, k - tol, -1e5, 1e5, k + tol, 1e5, dim=2),
                        0, Ly - k, 0, numElements=[1] * Ny, heights=h, recombine=True)

        if k < 0:
            h0_r = reverse_progression(h0)
            occ.extrude(occ.getEntitiesInBoundingBox(-r - tol, k-tol, -r - tol, r + tol, k+tol, r + tol, dim=2),
                        0, abs(k), 0, numElements=[1] * Nh, heights=h0_r, recombine=True)
            occ.synchronize()
            occ.extrude(occ.getEntitiesInBoundingBox(-1e5, -tol, -1e5, 1e5, tol, 1e5, dim=2),
                        0, abs(k), 0, numElements=[1] * Nh, heights=h0, recombine=True)
            occ.synchronize()
            occ.extrude(occ.getEntitiesInBoundingBox(-1e5, abs(k) - tol, -1e5, 1e5, abs(k) + tol, 1e5, dim=2),
                        0, Ly - abs(k), 0, numElements=[1] * Ny, heights=h, recombine=True)

        ### transfinite meshing controls
        occ.synchronize()

        # circles
        arcs = [
            otc, obc, olc, orc,
            ck_top, ck_bot, ck_left, ck_right,
            ci_top_left, ci_top_right, ci_bot_right, ci_bot_left,
            ca_top_left, ca_top_right, ca_bot_right, ca_bot_left
        ]

        rads = [circ_con_l, circ_con_t, circ_con_r, circ_con_b]
        oconns = [bla, bra, tla, tra]

        for arc in arcs: msh.setTransfiniteCurve(arc, N)
        for rad in rads: msh.setTransfiniteCurve(rad, Nr, coef=Pr)
        for oconn in oconns: msh.setTransfiniteCurve(oconn, Nri)

        # box lines
        for li in [top_left, top_right, right_top, right_bot, bot_right, bot_left, left_bot, left_top]:
            msh.setTransfiniteCurve(li, N)

        for li in [left_c, top_c, right_c, bot_c]:
            msh.setTransfiniteCurve(li, N // 2 + 1)

        # make all surfaces transfinite
        for S_i in [S_O, S_O1, S_O2, S_O3, S_O4, S_circ_tl, S_circ_tr, S_circ_br, S_circ_bl]:
            msh.setTransfiniteSurface(S_i)

        # meshing algorithm for non transfinite surfaces
        msh.setAlgorithm(2, S_TL, 8)
        msh.setAlgorithm(2, S_TR, 8)
        msh.setAlgorithm(2, S_BR, 8)
        msh.setAlgorithm(2, S_BL, 8)

        # mirror across z-axis: (x, y, z) -> (-x, y, z)
        A_x = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

        # mirror across x-axis: (x, y, z) -> (x, y, -z)
        A_z = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1]

        # 180° rotation around y-axis
        A_rot_y = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1]

        msh.setPeriodic(2, [S_BL], [S_BR], A_x)
        msh.setPeriodic(2, [S_TR], [S_BR], A_z)
        msh.setPeriodic(2, [S_TL], [S_BR], A_rot_y)

        vols = gmsh.model.getEntities(dim=3)

        top = gmsh.model.getEntitiesInBoundingBox(-Lin - tol, Ly - tol, -Lz - tol, Lx + tol, Ly + tol, Lz + tol, dim=2)
        inflow = gmsh.model.getEntitiesInBoundingBox(-Lin - tol, -tol, -Lz - tol, -Lin + tol, Ly + tol, Lz + tol, dim=2)
        outflow = gmsh.model.getEntitiesInBoundingBox(Lx - Lin - tol, -tol, -Lz - tol, Lx - Lin + tol, Ly + tol, Lz + tol, dim=2)
        left = gmsh.model.getEntitiesInBoundingBox(-Lin - tol, -tol, -Lz / 2 - tol, Lx - Lin + tol, Ly + tol, -Lz / 2 + tol, dim=2)
        right = gmsh.model.getEntitiesInBoundingBox(-Lin - tol, -tol, Lz / 2 - tol, Lx - Lin + tol, Ly + tol, Lz / 2 + tol, dim=2)

        if k > 0:
            pert = gmsh.model.getEntitiesInBoundingBox(-r - tol, -tol, -r - tol, r + tol, k + tol, r + tol, dim=2)
            plate = gmsh.model.getEntitiesInBoundingBox(-Lin - tol, -tol, -Lz - tol, Lx + tol, tol, Lz + tol, dim=2)

        if k < 0:
            hole = gmsh.model.getEntitiesInBoundingBox(-r - tol, -tol, -r - tol, r + tol, tol, r + tol, dim=2)
            pert = gmsh.model.getEntitiesInBoundingBox(-r - tol, k - tol, -r - tol, r + tol, tol, r + tol, dim=2)
            plate = gmsh.model.getEntitiesInBoundingBox(-Lin - tol, -tol, -Lz - tol, Lx + tol, tol, Lz + tol, dim=2)

            for dimtag in hole:
                if dimtag in pert: pert.remove(dimtag)
                if dimtag in plate: plate.remove(dimtag)

            # Remove fluid-fluid internal interfaces
            external_surfaces = set(gmsh.model.getBoundary(vols, combined=True, oriented=False, recursive=False))
            pert = [s for s in pert if s in external_surfaces]
            plate = [s for s in plate if s in external_surfaces]
            vols = gmsh.model.getEntities(dim=3)

        mod.add_physical_group(2, [_[1] for _ in inflow], name="inflow")
        mod.add_physical_group(2, [_[1] for _ in outflow], name="outflow")
        mod.add_physical_group(2, [_[1] for _ in left], name="left")
        mod.add_physical_group(2, [_[1] for _ in right], name="right")
        mod.add_physical_group(2, [_[1] for _ in top], name="top")
        mod.add_physical_group(2, [_[1] for _ in plate], name="plate")
        mod.add_physical_group(2, [_[1] for _ in pert], name="pert")
        mod.add_physical_group(3, [_[1] for _ in vols], name="volume")

        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Renumber", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.SaveAll", 0)

        msh.generate(3)
        msh.setOrder(2)
        msh.removeDuplicateNodes()
        gmsh.write(str(self.mshfile))

        element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
        self.Nelem = sum(len(tags) for tags in element_tags)

        # store actual mesh parameters
        self.N = N
        self.Nin = Nin
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Nh = Nh

        if save_config:
            config = {
                "outfile": outfile,
                "k": self.k,
                "eta": self.eta,
                "Lx": self.Lx,
                "Ly": self.Ly,
                "Lz": self.Lz,
                "Lin": self.Lin,
                "N": self.N,
                "Nin": self.Nin,
                "Nx": self.Nx,
                "Ny": self.Ny,
                "Nz": self.Nz,
                "Nh": self.Nh,
                "Nelem": self.Nelem
            }

            write_json(config, str(self.mshfile.with_suffix(".json")))

        if show:
            occ.synchronize()
            gmsh.fltk.run()
        gmsh.finalize()

        return self