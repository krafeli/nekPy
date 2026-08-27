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
        size_k=1/6, size_x=0.8, size_yk=0.12, size_y=0.75, size_z=None, size_in=1.20, d=None, STEP=0.1):

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

        self.size_k = size_k
        self.size_x = size_x
        self.size_yk = size_yk
        self.size_y = size_y
        self.size_z = self.size_x*0.6 if size_z is None else size_z
        self.size_in = size_in
        self.d = d
        self.STEP = STEP

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
            if self.Nelem is None: self.Nelem = config.get("Nelem")
            if self.size_k == 1/6: self.size_k = config.get("size_k", self.size_k)
            if self.size_x == 0.8: self.size_x = config.get("size_x", self.size_x)
            if self.size_yk == 0.12: self.size_yk = config.get("size_yk", self.size_yk)
            if self.size_y == 0.75: self.size_y = config.get("size_y", self.size_y)
            if self.size_z is None: self.size_z = config.get("size_z", self.size_x*0.6)
            if self.size_in == 1.20: self.size_in = config.get("size_in", self.size_in)
            if self.d is None: self.d = config.get("d")

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
            f"  Nelem = {self.Nelem}\n"
            f")"
        )

    # ---------------------------------------------------------------------------
    # Choose d automatically.
    # d defines the square surrounding the outer circle.  I search over possible
    # values and choose the one for which the minimum mesh qualitiy is maximized.
    # ---------------------------------------------------------------------------
    def _candidate_values(self, ra, size_k):
        """Return d candidates from smallest to largest."""
        d_min = ra + 0.5 * size_k
        d_max = ra + 5.0 * size_k
        step = self.STEP * size_k
        return np.arange(d_min, d_max + 0.5 * step, step)

    def _run_d_probe(self, dc):
        """
        Run this mesh in probe mode with a fixed d.  The probe
        generates the 2D mesh and returns the post-symmetry quality.
        """
        probe = Mesh(
            outfile="probe.msh",
            k=self.k,
            eta=self.eta,
            Lx=self.Lx,
            Ly=self.Ly,
            Lz=self.Lz,
            Lin=self.Lin,
            size_k=self.size_k,
            size_x=self.size_x,
            size_yk=self.size_yk,
            size_y=self.size_y,
            size_z=self.size_z,
            size_in=self.size_in,
            d=dc
        )
        try: return probe.generate(probe=True)
        except Exception as e:
            print(f"d={dc:.6g}: probe failed: {e}")
            return None

    def _optimize_d(self, ra):
        """
        Search d from small to large and choose the d whose actual 2D mesh has the
        best minimum quality.
        """
        results = []
        for dc in self._candidate_values(ra, self.size_k):
            q = self._run_d_probe(float(dc))
            if q is not None and np.isfinite(q[0]): results.append((float(dc), q[0], q[1]))

        if not results: raise RuntimeError("No candidate d produced a valid 2D mesh.")
        # Best achievable minimum quality in the search.
        best_q = max(r[1] for r in results)
        chosen = max(results, key=lambda r: r[1])
        d = chosen[0]
        print(f"Selected d={d:.6g}: post-symmetry qmin={chosen[1]:.6g}, qmean={chosen[2]:.6g}; best qmin={best_q:.6g}")
        return d

    def generate(self, show=False, save_config=True, probe=False):

        if self.k is None: raise ValueError("k is required for mesh generation")
        if self.eta is None: raise ValueError("eta is required for mesh generation")
        if self.Lx is None: raise ValueError("Lx is required for mesh generation")
        if self.Ly is None: raise ValueError("Ly is required for mesh generation")
        if self.Lz is None: raise ValueError("Lz is required for mesh generation")

        outfile = str(self.outfile)
        k, eta = self.k, self.eta
        Lx, Ly, Lz, Lin = self.Lx, self.Ly, self.Lz, self.Lin

        # dimensions
        dk = abs(eta * k)                       # cylinder diameter
        rk = dk/2.                              # cylinder radius
        ra = rk + 3*self.size_k                 # outer circle radius, chosen to capture 3 elements of size_k

        # ---------------------------------------------------------------------------
        # Choose d automatically.
        # d defines the square surrounding the outer circle.  I search over possible
        # values and choose the one for which the minimum mesh qualitiy is maximized.
        # ---------------------------------------------------------------------------
        if self.d is None:
            d = self._optimize_d(ra)
        else:
            d = self.d

        # In probe mode, d is supplied by the parent process. Otherwise search for d
        self.d = d
        Lx = Lx + Lin

        Nk = int(dk / self.size_k) - int(dk)
        Nx = int(Lx / self.size_x) - 1
        Ny = int(self.Ly / self.size_y) - 1
        Nz = int(Lz/2 / self.size_z) - 1
        Nin = int(Lin / self.size_in) - 1
        Nh = int(abs(k) / self.size_yk) - 1
        Nk = Nk - 1 if Nk % 2 == 0 else Nk      # Nk needs to be odd
        Nk = max(Nk, 3)
        Nr = int((ra-rk) / self.size_k) + 2

        # Progressions
        Pr = 1.25                               # radial circle progression
        Pw = 1.2                                # progression at wall until y=k

        dd = d / (3 + 1.5*dk)
        dh = k / Nh
        # progression ratios are chosen so the first element matches the target sizes
        Px = progression_ratio(Lx - Lin - d, Nx - 1, dd)
        Pz = progression_ratio((Lz - d) / 2.0, Nz - 1, dd)
        Py = progression_ratio(self.Ly - k, Ny - 1, dh)
        Pin = progression_ratio(Lin - d, Nin - 1, dd)

        gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)

        occ = gmsh.model.occ
        mod = gmsh.model
        point, line, carc = occ.addPoint, occ.addLine, occ.addCircleArc
        cloop, psurf = occ.addCurveLoop, occ.addPlaneSurface
        getBB = occ.getEntitiesInBoundingBox

        pi, sin, cos, sqrt = math.pi, math.sin, math.cos, math.sqrt
        msh = gmsh.model.mesh

        ##################################################################################
        ### cylinder circle

        cm = point(0,0,0)
        ci1 = point(rk, 0, 0)
        ci2 = point(-rk, 0, 0)
        cimt = point(0, 0, -rk)
        cimb = point(0, 0, rk)
        ci_top_left = carc(ci2, point(-rk/sqrt(2), 0, -rk/sqrt(2)), cimt, center=False)
        ci_top_right = carc(cimt, point(rk/sqrt(2), 0, -rk/sqrt(2)), ci1, center=False)
        ci_bot_right = carc(ci1, point(rk/sqrt(2), 0, rk/sqrt(2)), cimb, center=False)
        ci_bot_left = carc(cimb, point(-rk/sqrt(2), 0, rk/sqrt(2)), ci2, center=False)

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
        ks = math.copysign(k, eta)
        ckm = point(0, ks, 0)

        # cicle points and arcs
        cktr = point(rk*cos(-pi/4),   ks, rk*sin(-pi/4))
        cktl = point(rk*cos(-3*pi/4), ks, rk*sin(-3*pi/4))
        ckbr = point(rk*cos(pi/4),    ks, rk*sin(pi/4))
        ckbl = point(rk*cos(3*pi/4),  ks, rk*sin(3*pi/4))
        ck_top = carc(cktl, ckm, cktr)
        ck_bot = carc(ckbr, ckm, ckbl)
        ck_left = carc(ckbl, ckm, cktl)
        ck_right = carc(cktr, ckm, ckbr)

        # O-grid
        lam = 0.75                                          # smaller = more curved o grid
        rb = 0.75 * rk                                      # box radius
        tcp = point(0, ks, -lam*rk)
        bcp = point(0, ks, lam*rk)
        lcp = point(-lam*rk, ks, 0)
        rcp = point(lam*rk, ks, 0)
        otr = point(rb*cos(-pi/4), ks, rb*sin(-pi/4))
        otl = point(rb*cos(-3*pi/4), ks, rb*sin(-3*pi/4))
        obr = point(rb*cos(pi/4), ks, rb*sin(pi/4))
        obl = point(rb*cos(3*pi/4), ks, rb*sin(3*pi/4))
        otc = carc(otl, bcp, otr)
        obc = carc(obl, tcp, obr)
        olc = carc(obl, rcp, otl)
        orc = carc(obr, lcp, otr)

        # arc-cirle connection
        tla = line(cktl, otl)
        tra = line(cktr, otr)
        bra = line(ckbr, obr)
        bla = line(ckbl, obl)

        # surfaces
        # O-Surface
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

        Rt = add_rect(-d, d, -d, -Lz / 2, nx=2 * Nk - 1, nz=Nz, pz=Pz)                        # top rectangle
        Rb = add_rect(-d, d, d, Lz / 2, nx=2 * Nk - 1, nz=Nz, pz=Pz)                          # bottom rectangle
        Rint = add_rect(-Lin, -d, -Lz / 2, -d, nx=Nin, nz=Nz, px=-Pin, pz=-Pz)                # top inflow rect
        Rm = add_rect(-Lin, -d, -d, d, nx=Nin, nz=2 * Nk - 1, px=-Pin)                        # mid inflow rect
        Rinb = add_rect(-Lin, -d, d, Lz / 2, nx=Nin, nz=Nz, px=-Pin, pz=Pz)                   # bottom inflow rect
        Routt = add_rect(d, Lx - Lin, -Lz / 2, -d, nx=Nx, nz=Nz, px=Px, pz=-Pz)
        Routm = add_rect(d, Lx - Lin, -d, d, nx=Nx, nz=2 * Nk - 1, px=Px)
        Routb = add_rect(d, Lx - Lin, d, Lz / 2, nx=Nx, nz=Nz, px=Px, pz=Pz)

        # mesh extrusion
        tol = 1e-3
        h0 = progression(Nh, Pw)
        h = progression(Ny, Py)

        if eta > 0:
            occ.extrude(getBB(-Lin - tol, -tol, -Lz / 2 - tol, Lx - Lin + tol, tol, Lz / 2 + tol, dim=2), 0, k, 0, numElements=[1] * Nh, heights=h0, recombine=True)
            occ.synchronize()
            occ.extrude(getBB(-1e5, k - tol, -1e5, 1e5, k + tol, 1e5, dim=2), 0, Ly - k, 0, numElements=[1] * Ny, heights=h, recombine=True)
        if eta < 0:
            h0_r = reverse_progression(h0)
            occ.extrude(getBB(-rk - tol, ks - tol, -rk - tol, rk + tol, ks + tol, rk + tol, dim=2), 0, k, 0, numElements=[1] * Nh, heights=h0_r, recombine=True)
            occ.synchronize()
            occ.extrude(getBB(-1e5, -tol, -1e5, 1e5, tol, 1e5, dim=2), 0, k, 0, numElements=[1] * Nh, heights=h0, recombine=True)
            occ.synchronize()
            occ.extrude(getBB(-1e5, k - tol, -1e5, 1e5, k + tol, 1e5, dim=2), 0, Ly - k, 0, numElements=[1] * Ny, heights=h, recombine=True)

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

        for arc in arcs: msh.setTransfiniteCurve(arc, Nk)
        for rad in rads: msh.setTransfiniteCurve(rad, Nr, coef=Pr)
        for oconn in oconns: msh.setTransfiniteCurve(oconn, Nk // 2)

        # box lines
        for li in [top_left, top_right, right_top, right_bot, bot_right, bot_left, left_bot, left_top]:     # outer lines
            msh.setTransfiniteCurve(li, Nk)
        for li in [left_c, top_c, right_c, bot_c]:
            msh.setTransfiniteCurve(li, 3)

        # make all surfaces transfinite
        for S_i in [S_O, S_O1, S_O2, S_O3, S_O4, S_circ_tl, S_circ_tr, S_circ_br, S_circ_bl]:
            msh.setTransfiniteSurface(S_i)

        # meshing algorithm for non transfinite surfaces
        msh.setAlgorithm(2, S_TL, 8)
        msh.setAlgorithm(2, S_TR, 8)
        msh.setAlgorithm(2, S_BR, 8)
        msh.setAlgorithm(2, S_BL, 8)

        # Periodic symmetry is applied after the mesh has been generated. The Quadrant with the best min quality is chosen
        # Affine transformations mapping each quadrant -> BR.
        A_x = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        A_z = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1]
        A_rot_y = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 1]
        quadrants = {"BR": S_BR, "BL": S_BL, "TR": S_TR, "TL": S_TL}

        def _surface_quality(surface):
            _, elem_tags, _ = msh.getElements(2, surface)
            tags = np.concatenate([np.asarray(tt, dtype=int) for tt in elem_tags if len(tt)])
            if len(tags) == 0: return float("-inf"), float("-inf")
            q = np.asarray(msh.getElementQualities(tags), dtype=float)
            return float(np.min(q)), float(np.mean(q))

        msh.generate(2)
        qualities = {name: _surface_quality(surface) for name, surface in quadrants.items()}
        best_quadrant = max(quadrants, key=lambda name: qualities[name])
        # Clear the temporary mesh and apply periodicity using the selected quadrant as master.
        msh.clear()

        if best_quadrant == "BR":
            msh.setPeriodic(2, [S_BL], [S_BR], A_x)
            msh.setPeriodic(2, [S_TR], [S_BR], A_z)
            msh.setPeriodic(2, [S_TL], [S_BR], A_rot_y)
        elif best_quadrant == "BL":
            msh.setPeriodic(2, [S_BR], [S_BL], A_x)
            msh.setPeriodic(2, [S_TL], [S_BL], A_z)
            msh.setPeriodic(2, [S_TR], [S_BL], A_rot_y)
        elif best_quadrant == "TR":
            msh.setPeriodic(2, [S_TL], [S_TR], A_x)
            msh.setPeriodic(2, [S_BR], [S_TR], A_z)
            msh.setPeriodic(2, [S_BL], [S_TR], A_rot_y)
        elif best_quadrant == "TL":
            msh.setPeriodic(2, [S_TR], [S_TL], A_x)
            msh.setPeriodic(2, [S_BL], [S_TL], A_z)
            msh.setPeriodic(2, [S_BR], [S_TL], A_rot_y)

        # In probe mode, I now generate the actual 2D mesh, with symmetry constraint applied.
        if probe:
            msh.generate(2)
            post_qualities = {name: _surface_quality(surface) for name, surface in quadrants.items()}
            qmin = min(q[0] for q in post_qualities.values())
            qmean = min(q[1] for q in post_qualities.values())
            print(f"d={d:.6g}: qmin={qmin:.6g}, qmean={qmean:.6g}")
            gmsh.finalize()
            return qmin, qmean

        getEBB = gmsh.model.getEntitiesInBoundingBox
        vols = gmsh.model.getEntities(dim=3)
        top = getEBB(-Lin - tol, Ly - tol, -Lz - tol, Lx + tol, Ly + tol, Lz + tol, dim=2)
        inflow = getEBB(-Lin - tol, -tol, -Lz - tol, -Lin + tol, Ly + tol, Lz + tol, dim=2)
        outflow = getEBB(Lx - Lin - tol, -tol, -Lz - tol, Lx - Lin + tol, Ly + tol, Lz + tol, dim=2)
        left = getEBB(-Lin - tol, -tol, -Lz / 2 - tol, Lx - Lin + tol, Ly + tol, -Lz / 2 + tol, dim=2)
        right = getEBB(-Lin - tol, -tol, Lz / 2 - tol, Lx - Lin + tol, Ly + tol, Lz / 2 + tol, dim=2)

        if eta >= 0:
            pert = getEBB(-rk - tol, 0 - tol, -rk - tol, rk + tol, k + tol, rk + tol, dim=2)
            plate = getEBB(-Lin - tol, 0 - tol, -Lz - tol, Lx + tol, 0 + tol, Lz + tol, dim=2)
        elif eta < 0:
            hole = getEBB(-rk - tol, -tol, -rk - tol, rk + tol, tol, rk + tol, dim=2)
            pert = getEBB(-rk - tol, ks - tol, -rk - tol, rk + tol, tol, rk + tol, dim=2)
            plate = getEBB(-Lin - tol, -tol, -Lz - tol, Lx + tol, tol, Lz + tol, dim=2)
            for dimtag in hole:
                if dimtag in pert:
                    pert.remove(dimtag)
                if dimtag in plate:
                    plate.remove(dimtag)

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

        if save_config:
            config = {
                "outfile": outfile,
                "k": self.k,
                "eta": self.eta,
                "Lx": self.Lx,
                "Ly": self.Ly,
                "Lz": self.Lz,
                "Lin": self.Lin,
                "Nelem": self.Nelem,
                "size_k": self.size_k,
                "size_x": self.size_x,
                "size_yk": self.size_yk,
                "size_y": self.size_y,
                "size_z": self.size_z,
                "size_in": self.size_in,
                "d": self.d,
                "dd": dd,
                "Nr": Nr,
                "Pin": Pin,
                "Pout": Px,
                "Pri": 1.25,
                "Py": Py,
                "Pz": Pz,
                "ra": ra
            }

            write_json(config, str(self.mshfile.with_suffix(".json")))

        if show:
            occ.synchronize()
            gmsh.fltk.run()
        gmsh.finalize()

        return self

