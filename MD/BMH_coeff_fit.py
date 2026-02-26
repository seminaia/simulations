"""
Born-Mayer-Huggins potential fitting using GPAW DFT
====================================================
Fits BMH short-range parameters (A, ρ, C, D) for each ion pair
in LiF·BeF₂+H by scanning dimer energies with GPAW PBE/PW and
subtracting the point-charge Coulomb term.

  Full BMH potential (LAMMPS born/coul/long):
      V(r) = A·exp((σ−r)/ρ) − C/r⁶ + D/r⁸   +   k·q₁·q₂/r
             └──────────── short-range ───────┘   └─ Ewald ─┘

  pair_coeff i j  A(eV)  ρ(Å)  σ(Å)  C(eV·Å⁶)  D(eV·Å⁸)
    σ = contact/collision diameter (Å); sets the energy scale of the repulsion

Method (reference-subtraction — avoids monomer spin issues):
  E_sr(r) = [E_dimer(r) − E_dimer(r_ref)]
            − k·q₁·q₂·(1/r − 1/r_ref)

  where r_ref is large enough that E_sr(r_ref) ≈ 0.

Dispersion fitting tiers:
  F–F           → fit A, ρ, C, D   (dominant anion–anion dispersion)
  H–F           → fit A, ρ, C      (C₆ expected; D small for a proton)
  Li–F, Be–F    → fit A, ρ         (cation polarisability negligible)

Cation–cation pairs (Li–Li, Li–Be, Be–Be, H–H, H–Li, H–Be)
are purely Coulombic → A = C = D = 0 in LAMMPS.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import epsilon_0, e, physical_constants 
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac

# ── Configuration ──────────────────────────────────────────────────────────────

# Ionic charges (must match LAMMPS charge_map in classical_md.py)
CHARGES = {'Li': +1.0, 'Be': +2.0, 'F': -1.0, 'H': +1.0}

# LAMMPS type ordering — must match specorder in classical_md.py
SPECORDER = ['Li', 'Be', 'F', 'H']

# Pairs that need short-range DFT scanning.
# H⁺ (proton, q=+1): only H–F has meaningful short-range interaction.
# All cation–cation pairs are purely Coulombic → not listed here.
SCAN_PAIRS = [
    ('Li', 'F'),
    ('Be', 'F'),
    ('F',  'F'),
    ('H',  'F'),
]

# Dispersion fitting tiers (pairs not listed get C = D = 0)
FIT_CD = {('F', 'F')}          # fit A, ρ, C, D  — full BMH
FIT_C  = {('H', 'F')}          # fit A, ρ, C     — D=0

# Contact distance σ (Å): minimum r below which LAMMPS holds E constant.
# Set to a safe value slightly below R_MIN; not part of the energy expression.
SIGMA_DEFAULT = 1.0   # Å

# r grid for dimer scan
R_MIN = 0.8    # Å  — avoid core-core divergence in DFT
R_MAX = 5.0    # Å  — well into the flat / zero region
N_R   = 10     # number of separation points
R_REF = 8.0    # Å  — large-separation reference; E_sr(r_ref) ≈ 0

# GPAW plane-wave settings
ECUT_EV = 400   # eV
VACUUM  = 7.0   # Å vacuum on each side of the dimer

# Coulomb constant  k_e·e²  in eV·Å
K_COULOMB = e * 1e10 / (4 * np.pi * epsilon_0)  # F/m
# ── GPAW helpers ──────────────────────────────────────────────────────────────
def make_gpaw(txt='-'):
    """GPAW PW calculator for isolated dimers (Γ-point, mild smearing)."""
    return GPAW(
        mode        = PW(ECUT_EV),
        xc          = 'PBE',
        kpts        = {'size': (1, 1, 1), 'gamma': True},
        occupations = FermiDirac(0.05),
        symmetry    = 'off',
        txt         = txt,
        convergence = {'energy': 1e-5},
        maxiter     = 500,
    )


def dimer_atoms(sym1, sym2, r, vacuum=VACUUM):
    """Neutral atom dimer: sym1 at origin, sym2 at (r,0,0), in vacuum box."""
    atoms = Atoms([sym1, sym2], positions=[(0, 0, 0), (r, 0, 0)])
    atoms.center(vacuum=vacuum)
    return atoms


def get_energy(atoms, log_tag):
    """Attach a fresh GPAW calculator and return the potential energy (eV)."""
    atoms.calc = make_gpaw(txt=f'gpaw_{log_tag}.log')
    return atoms.get_potential_energy()


# ── Dimer energy scan ─────────────────────────────────────────────────────────

def scan_pair(sym1, sym2, r_values):
    """
    Return the short-range BMH energy E_sr(r) at each separation.

    E_sr(r) = [E_dimer(r) − E_dimer(r_ref)] − k·q₁·q₂·(1/r − 1/r_ref)

    The reference subtraction cancels both monomer energies and avoids
    spin-polarised monomer DFT calculations.
    """
    q1  = CHARGES.get(sym1, 0.0)
    q2  = CHARGES.get(sym2, 0.0)
    tag = f'{sym1}{sym2}'

    print(f"  Reference  r_ref = {R_REF:.1f} Å ...", end='  ', flush=True)
    e_ref = get_energy(dimer_atoms(sym1, sym2, R_REF), f'{tag}_ref')
    print(f"E_ref = {e_ref:.6f} eV")

    e_sr_list = []
    for r in r_values:
        e_dim      = get_energy(dimer_atoms(sym1, sym2, r), f'{tag}_r{r:.2f}')
        delta_e    = e_dim - e_ref
        delta_coul = K_COULOMB * q1 * q2 * (1.0 / r - 1.0 / R_REF)
        e_sr       = delta_e - delta_coul
        e_sr_list.append(e_sr)
        print(f"    r={r:.2f} Å  ΔE={delta_e:+.4f}  "
              f"ΔE_coul={delta_coul:+.4f}  E_sr={e_sr:+.4f} eV")

    return np.array(e_sr_list)


# ── BMH functional forms ──────────────────────────────────────────────────────

def bmh(r, A, sigma, rho, C, D):
    """Full BMH short-range: A·exp((σ−r)/ρ) − C/r⁶ + D/r⁸"""
    return A * np.exp((sigma - r) / rho) - C / r**6 + D / r**8


def bmh_C(r, A, sigma, rho, C):
    """BMH without r⁻⁸ term: A·exp((σ−r)/ρ) − C/r⁶"""
    return A * np.exp((sigma - r) / rho) - C / r**6


def bmh_rep(r, A, sigma, rho):
    """Pure repulsion: A·exp((σ−r)/ρ)"""
    return A * np.exp((sigma - r) / rho)


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_bmh(sym1, sym2, r_values, e_sr):
    """
    Fit BMH parameters to E_sr(r).

    Points excluded from the fit:
      - r < 1.8 Å   (DFT core region may be unreliable)
      - E_sr > 20 eV (likely unconverged SCF at very short r)
      - E_sr < −2 eV (unphysical attraction — numerical noise)

    Returns:
      params  : np.ndarray [A, sigma, rho, C, D]  in eV, Å, Å, eV·Å⁶, eV·Å⁸
      err     : np.ndarray  1-σ uncertainties (nan if fit failed)
    """
    pair     = (sym1, sym2)
    pair_rev = (sym2, sym1)
    do_CD = pair in FIT_CD or pair_rev in FIT_CD
    do_C  = pair in FIT_C  or pair_rev in FIT_C

    mask  = (r_values >= 1.8) & (e_sr < 20.0) & (e_sr > -2.0)
    r_fit = r_values[mask]
    e_fit = e_sr[mask]

    if len(r_fit) < 4:
        print("    WARNING: fewer than 4 usable points — returning zeros.")
        return np.zeros(5), np.full(5, np.nan)

    try:
        if do_CD:
            popt, pcov = curve_fit(
                bmh, r_fit, e_fit,
                p0     = [500.0, 1.0, 0.25, 5.0, 2.0],
                bounds = ([0, 0.01, 0.01, 0, 0], [1e6, 5.0, 2.0, 500.0, 500.0]),
                maxfev = 100_000,
            )
            A, sigma, rho, C, D = popt
            err = np.sqrt(np.diag(pcov))

        elif do_C:
            popt, pcov = curve_fit(
                bmh_C, r_fit, e_fit,
                p0     = [300.0, 1.0, 0.25, 2.0],
                bounds = ([0, 0.01, 0.01, 0], [1e6, 5.0, 2.0, 500.0]),
                maxfev = 100_000,
            )
            A, sigma, rho, C = popt
            D   = 0.0
            err = np.append(np.sqrt(np.diag(pcov)), 0.0)

        else:
            popt, pcov = curve_fit(
                bmh_rep, r_fit, e_fit,
                p0     = [300.0, 1.0, 0.25],
                bounds = ([0, 0.01, 0.01], [1e6, 5.0, 2.0]),
                maxfev = 100_000,
            )
            A, sigma, rho = popt
            C, D   = 0.0, 0.0
            err    = np.append(np.sqrt(np.diag(pcov)), [0.0, 0.0])

    except RuntimeError as exc:
        print(f"    WARNING: curve_fit failed: {exc}")
        return np.zeros(5), np.full(5, np.nan)

    return np.array([A, sigma, rho, C, D]), err


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    r_values = np.linspace(R_MIN, R_MAX, N_R)
    results  = {}

    # ── scan + fit ────────────────────────────────────────────────────────────
    for sym1, sym2 in SCAN_PAIRS:
        print(f"\n{'='*60}")
        print(f"Pair  {sym1}–{sym2}  "
              f"(q₁={CHARGES.get(sym1,0):+.0f}, q₂={CHARGES.get(sym2,0):+.0f})")
        print(f"{'='*60}")

        e_sr               = scan_pair(sym1, sym2, r_values)
        params, err        = fit_bmh(sym1, sym2, r_values, e_sr)
        A, sigma, rho, C, D = params

        r_dense = np.linspace(R_MIN, R_MAX, 300)
        e_model = bmh(r_dense, A, sigma, rho, C, D)

        results[(sym1, sym2)] = {
            'A': A, 'sigma': sigma, 'rho': rho, 'C': C, 'D': D,
            'err': err,
            'r': r_values, 'e_sr': e_sr,
            'r_dense': r_dense, 'e_model': e_model,
        }

        print(f"\n  ── Fitted Born-Mayer-Huggins parameters ──────────────────")
        print(f"     A   = {A:10.4f}  ± {err[0]:.4f}   eV")
        print(f"     σ   = {sigma:10.5f}  ± {err[1]:.5f}   Å")
        print(f"     ρ   = {rho:10.5f}  ± {err[2]:.5f}   Å")
        print(f"     C   = {C:10.4f}  ± {err[3]:.4f}   eV·Å⁶")
        print(f"     D   = {D:10.4f}  ± {err[4]:.4f}   eV·Å⁸")

    # ── LAMMPS pair_coeff table ───────────────────────────────────────────────
    # pair_style born/coul/long: pair_coeff i j  A  rho  sigma  C  D
    print(f"\n{'='*60}")
    print("LAMMPS  pair_coeff  (pair_style born/coul/long)")
    print(f"{'='*60}")
    print(f"  # {'i':>2} {'j':>2}  {'A (eV)':>10}  {'rho (Å)':>9}  "
          f"{'sigma (Å)':>9}  {'C (eV·Å⁶)':>12}  {'D (eV·Å⁸)':>12}  pair")

    pair_coeff_lines = []
    for i, s1 in enumerate(SPECORDER, 1):
        for j, s2 in enumerate(SPECORDER[i-1:], i):
            fwd = (s1, s2)
            rev = (s2, s1)
            dat = results.get(fwd) or results.get(rev)
            if dat:
                A   = dat['A']
                sig = dat['sigma']
                rho = dat['rho']
                C   = dat['C']
                D   = dat['D']
                line = (f"pair_coeff {i} {j}  {A:10.4f}  {rho:.5f}  "
                        f"{sig:.5f}  {C:10.4f}  {D:10.4f}   # {s1}–{s2}")
            else:
                line = (f"pair_coeff {i} {j}      0.0000  1.00000  "
                        f"{SIGMA_DEFAULT:.5f}      0.0000      0.0000"
                        f"   # {s1}–{s2}  (Coulomb only)")
            print(f"  {line}")
            pair_coeff_lines.append(line)

    # ── write to file ─────────────────────────────────────────────────────────
    out_file = 'BMH_pair_coeff.txt'
    with open(out_file, 'w') as fh:
        fh.write("# Born-Mayer-Huggins pair coefficients\n")
        fh.write("# Fitted from GPAW/PBE dimer energy scans\n")
        fh.write(f"# r scan: {R_MIN:.1f}–{R_MAX:.1f} Å  ({N_R} points)\n")
        fh.write(f"# r_ref:  {R_REF:.1f} Å   E_cut: {ECUT_EV} eV   xc: PBE\n")
        fh.write("# pair_style born/coul/long\n")
        fh.write("# Format: pair_coeff i j  A(eV)  rho(Å)  sigma(Å)  C(eV·Å⁶)  D(eV·Å⁸)\n")
        fh.write("#\n")
        for ln in pair_coeff_lines:
            fh.write(ln + "\n")
    print(f"\nPair coefficients written → {out_file}")

    # ── Python list for classical_md.py ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Python  pair_coeff  list  (pair_style born/coul/long)")
    print(f"{'='*60}")
    print("pair_coeff = [")
    for i, s1 in enumerate(SPECORDER, 1):
        for j, s2 in enumerate(SPECORDER[i-1:], i):
            fwd = (s1, s2)
            rev = (s2, s1)
            dat = results.get(fwd) or results.get(rev)
            if dat:
                A, sigma, rho, C, D = dat['A'], dat['sigma'], dat['rho'], dat['C'], dat['D']
                print(f"    '{i} {j}  {A:10.4f}  {rho:.5f}  {sigma:.5f}  "
                      f"{C:10.4f}  {D:10.4f}',   # {s1}–{s2}")
            else:
                print(f"    '{i} {j}      0.0000  1.00000  {SIGMA_DEFAULT:.5f}  "
                      f"     0.0000       0.0000',   # {s1}–{s2}  Coulomb only")
    print("]")

    # ── plot ──────────────────────────────────────────────────────────────────
    n  = len(results)
    nc = min(2, n)
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 4*nr), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ((s1, s2), dat) in zip(axes_flat, results.items()):
        A, sigma, rho, C, D = dat['A'], dat['sigma'], dat['rho'], dat['C'], dat['D']
        ax.scatter(dat['r'], dat['e_sr'], color='steelblue', s=50,
                   zorder=3, label='GPAW PBE')
        ax.plot(dat['r_dense'], dat['e_model'], 'r-', lw=1.8,
                label=(f"BMH fit\n"
                       f"A={A:.1f} σ={sigma:.3f} ρ={rho:.3f}\n"
                       f"C={C:.2f} D={D:.2f}"))
        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.set_xlabel('r (Å)')
        ax.set_ylabel('$E_{sr}$ (eV)')
        ax.set_title(f'{s1}–{s2}')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ylo = max(dat['e_sr'].min() - 0.5, -3.0)
        yhi = min(dat['e_sr'].max() + 0.5, 15.0)
        ax.set_ylim(ylo, yhi)

    for ax in axes_flat[n:]:
        ax.axis('off')

    plt.suptitle('Born-Mayer-Huggins fit  (GPAW/PBE dimer scans)', fontsize=12)
    plt.tight_layout()
    plot_file = 'BMH_fit_results.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {plot_file}")


if __name__ == '__main__':
    main()
