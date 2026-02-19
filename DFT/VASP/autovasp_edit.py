# === Drop-in: magnetic order from auto_vasp.in =================================
# In auto_vasp.in, add e.g.:
#   MAGNETIC_ORDER = AFM-G         # one of: NM, FM, AFM, AFM-A, AFM-C, AFM-G
#   MAGNETIC       = Fe 4.0 Co 3.0 Ni 2.0   # element–moment pairs (μB)
#
# Usage in your flow:
#   mag = MagneticOrderDropIn()           # reads auto_vasp.in + POSCAR
#   incar = mag.apply_to_incar(incar)     # sets ISPIN/MAGMOM (removes MAGMOM if NM)
# ==============================================================================

from typing import Dict, List, Tuple
import math

class MagneticOrderDropIn:
    def __init__(self, infile: str = "auto_vasp.in", poscar: str = "POSCAR"):
        self.infile = infile
        self.poscar = poscar
        cfg = self._parse_infile(infile)
        order = cfg.get("MAGNETIC_ORDER", "NM").strip().upper()
        if order == "AFM":  # treat bare AFM as G-type
            order = "AFM-G"
        self.order: str = order
        self.elem_mags: Dict[str, float] = self._parse_pairs(cfg.get("MAGNETIC", ""))

    # ---- public API ----
    def incar_updates(self) -> Dict[str, str]:
        if self.order == "NM" or not self.elem_mags:
            return {"ISPIN": "1"}  # NM: ensure no MAGMOM is written
        species, frac = self._parse_poscar(self.poscar)
        magmom = self._build_magmom(species, frac)
        return {"ISPIN": "2", "MAGMOM": " ".join(f"{m:g}" for m in magmom)}

    def apply_to_incar(self, incar: Dict[str, object]) -> Dict[str, object]:
        upd = self.incar_updates()
        if upd.get("ISPIN") == "1":
            incar["ISPIN"] = "1"
            incar.pop("MAGMOM", None)
        else:
            incar.update(upd)
        return incar

    # ---- helpers: parse infile ----
    @staticmethod
    def _parse_infile(path: str) -> Dict[str, str]:
        cfg: Dict[str, str] = {}
        try:
            with open(path, "r", errors="ignore") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        cfg[k.strip().upper()] = v.strip()
        except FileNotFoundError:
            pass
        return cfg

    @staticmethod
    def _parse_pairs(s: str) -> Dict[str, float]:
        if not s:
            return {}
        toks = s.split()
        if len(toks) % 2 != 0:
            raise ValueError(f"Bad MAGNETIC string: {s!r}")
        return {toks[i]: float(toks[i+1]) for i in range(0, len(toks), 2)}

    # ---- POSCAR (VASP 5+) -> species + fractional coords ----
    @staticmethod
    def _is_ints(tokens: List[str]) -> bool:
        try:
            _ = [int(t) for t in tokens]; return True
        except Exception:
            return False

    def _parse_poscar(self, path: str) -> Tuple[List[str], List[Tuple[float,float,float]]]:
        with open(path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        scale = float(lines[1])
        a = [float(x) for x in lines[2].split()[:3]]
        b = [float(x) for x in lines[3].split()[:3]]
        c = [float(x) for x in lines[4].split()[:3]]
        A = [[scale*a[0], scale*a[1], scale*a[2]],
             [scale*b[0], scale*b[1], scale*b[2]],
             [scale*c[0], scale*c[1], scale*c[2]]]
        i = 5
        toks = lines[i].split()
        if self._is_ints(toks):
            raise ValueError("POSCAR lacks element symbols line (need VASP5 format).")
        symbols = toks; i += 1
        counts = [int(x) for x in lines[i].split()]; i += 1
        if lines[i].lower().startswith("s"):  # Selective Dynamics
            i += 1
        mode = lines[i].lower()[0]; i += 1  # 'd' or 'c'
        nsite = sum(counts)
        species = [sym for sym, cnt in zip(symbols, counts) for _ in range(cnt)]
        coords = []
        for k in range(nsite):
            vals = [float(x) for x in lines[i+k].split()[:3]]
            coords.append(tuple(vals))
        if mode == "c":
            coords = [self._cart_to_frac(v, A) for v in coords]
        # Normalize to [0,1)
        frac = [tuple(v - math.floor(v) for v in p) for p in coords]
        return species, frac

    @staticmethod
    def _det3(M):
        return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])
              - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])
              + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))

    @staticmethod
    def _inv3(M):
        det = MagneticOrderDropIn._det3(M)
        if abs(det) < 1e-20:
            raise ValueError("Lattice matrix nearly singular.")
        cof = [
            [ (M[1][1]*M[2][2]-M[1][2]*M[2][1]),
             -(M[1][0]*M[2][2]-M[1][2]*M[2][0]),
              (M[1][0]*M[2][1]-M[1][1]*M[2][0])],
            [-(M[0][1]*M[2][2]-M[0][2]*M[2][1]),
              (M[0][0]*M[2][2]-M[0][2]*M[2][0]),
             -(M[0][0]*M[2][1]-M[0][1]*M[2][0])],
            [ (M[0][1]*M[1][2]-M[0][2]*M[1][1]),
             -(M[0][0]*M[1][2]-M[0][2]*M[1][0]),
              (M[0][0]*M[1][1]-M[0][1]*M[1][0])]
        ]
        return [[cof[j][i]/det for i in range(3)] for j in range(3)]

    def _cart_to_frac(self, v: Tuple[float,float,float], A: List[List[float]]):
        invA = self._inv3(A)
        x, y, z = v
        return (invA[0][0]*x + invA[0][1]*y + invA[0][2]*z,
                invA[1][0]*x + invA[1][1]*y + invA[1][2]*z,
                invA[2][0]*x + invA[2][1]*y + invA[2][2]*z)

    # ---- AFM patterns (flip every 1/2 cell) ----
    @staticmethod
    def _half_layer(frac: float) -> int:
        return int((frac / 0.5) + 1e-9) % 2

    def _sign_afm_g(self, fx: float, fy: float, fz: float) -> int:
        return +1 if (self._half_layer(fx) + self._half_layer(fy) + self._half_layer(fz)) % 2 == 0 else -1

    def _sign_afm_a(self, fz: float) -> int:
        return +1 if self._half_layer(fz) == 0 else -1

    def _sign_afm_c(self, fx: float, fy: float) -> int:
        return +1 if (self._half_layer(fx) + self._half_layer(fy)) % 2 == 0 else -1

    def _build_magmom(self, species: List[str], frac: List[Tuple[float,float,float]]) -> List[float]:
        moms: List[float] = []
        for (sym, (fx, fy, fz)) in zip(species, frac):
            m0 = self.elem_mags.get(sym, 0.0)
            if m0 == 0.0:
                moms.append(0.0); continue
            if self.order == "FM":
                sign = +1
            elif self.order == "AFM-G":
                sign = self._sign_afm_g(fx, fy, fz)
            elif self.order == "AFM-A":
                sign = self._sign_afm_a(fz)
            elif self.order == "AFM-C":
                sign = self._sign_afm_c(fx, fy)
            else:
                raise ValueError(f"Unknown MAGNETIC_ORDER: {self.order}")
            moms.append(sign * m0)
        return moms
