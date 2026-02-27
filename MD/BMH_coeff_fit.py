"""
Born-Mayer-Huggins potential fitting using GPAW DFT
====================================================
Fits BMH short-range parameters (A, ρ, C, D) for each ion pair
in LiF·BeF₂+H by scanning dimer energies with GPAW PBE/PW.

  Full BMH potential (LAMMPS born/coul/long):
      V(r) = A·exp((σ−r)/ρ) − C/r⁶ + D/r⁸   +   k·q₁·q₂/r
             └──────────── short-range ───────┘   └─ Ewald ─┘

  pair_coeff i j  A(eV)  ρ(Å)  σ(Å)  C(eV·Å⁶)  D(eV·Å⁸)
    σ = contact/collision diameter (Å); sets the energy scale of the repulsion

Method (reference subtraction):
  E_sr(r) = [E_DFT(r) − E_DFT(r_max)] − k·q₁q₂·(1/r − 1/r_max)

  Removes the ionic-state asymptote (E_DFT → -(IE−EA) ≠ 0 for Li-F, Be-F),
  so E_sr → 0 at r_max. Fit pure BMH to E_sr; parameters go directly to LAMMPS.
  σ is fixed from Shannon ionic radii; A is back-computed as A = B·exp(−σ/ρ).

Cation–cation pairs (Li–Li, Li–Be, Be–Be, H–H, H–Li, H–Be)
are purely Coulombic → A = C = D = 0 in LAMMPS.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import epsilon_0, e
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

# Contact distance σ (Å) per pair — sum of Shannon ionic radii (6-coord).
# σ is fixed during fitting; only B = A·exp(σ/ρ) and ρ are free parameters.
# After fitting, A is recovered via A = B·exp(−σ/ρ).
# Shannon radii (Å): Li⁺=0.76, Be²⁺=0.45, F⁻=1.33, H⁺≈0 (bare proton).
SIGMA_CONTACT = {
    ('Li', 'F'):  2.09,   # Li⁺(0.76) + F⁻(1.33)
    ('F',  'Li'): 2.09,
    ('Be', 'F'):  1.78,   # Be²⁺(0.45) + F⁻(1.33)
    ('F',  'Be'): 1.78,
    ('F',  'F'):  2.66,   # F⁻(1.33) + F⁻(1.33)
    ('H',  'F'):  1.33,   # H⁺(≈0) + F⁻(1.33)
    ('F',  'H'):  1.33,
}
SIGMA_DEFAULT = 1.0   # Å fallback for pairs not in SIGMA_CONTACT

# r grid for dimer scan
R_MIN = 1.0    # Å  — avoid core-core divergence in DFT
R_MAX = 5.0    # Å  — well into the flat / zero region
N_R   = 10     # number of separation points

# GPAW plane-wave settings
ECUT_EV = 400   # eV
VACUUM  = 7.0   # Å vacuum on each side of the dimer

# Coulomb constant  k_e  in eV·Å  (= e/(4πε₀) in SI, converted to eV·Å)
K_COULOMB = e * 1e10 / (4 * np.pi * epsilon_0)

# Relative dielectric constant ε_r (1 = vacuum; set >1 for screened Coulomb)
# Must match the 'dielectric' command in LAMMPS.
EPSILON_R = 1.0

# ── GPAW helpers ──────────────────────────────────────────────────────────────
def make_gpaw(txt='-'):
    """GPAW PW calculator for isolated dimers (Γ-point, mild smearing)."""
    return GPAW(
        mode        = PW(ECUT_EV),
        xc          = 'HSE06',
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
    Return the raw DFT potential energy E_pot(r) at each separation.

    GPAW references energies to isolated neutral atoms, so E_pot(r→∞) → 0.
    The Coulomb term is included directly in the fitting function (bmh_coul)
    rather than being pre-subtracted here.
    """
    q1  = CHARGES.get(sym1, 0.0)
    q2  = CHARGES.get(sym2, 0.0)
    tag = f'{sym1}{sym2}'

    e_pot_list = []
    for r in r_values:
        e_pot = get_energy(dimer_atoms(sym1, sym2, r), f'{tag}_r{r:.2f}')
        E_lr  = coul(q1, q2, r)
        e_sr  = e_pot - E_lr
        e_pot_list.append(e_pot)
        print(f"    r={r:.2f} Å  E_pot={e_pot:+.4f}  "
              f"E_lr={E_lr:+.4f}  E_sr={e_sr:+.4f} eV")
    return np.array(e_pot_list)

def coul(q1, q2, r):
    """Screened Coulomb energy: k·q₁·q₂ / (ε_r·r)"""
    return K_COULOMB * q1 * q2 / (EPSILON_R * r)

def lj(r, epsilon, sigma):
    """Lennard-Jones: 4ε[(σ/r)¹² − (σ/r)⁶]"""
    sr6 = (sigma / r)**6
    return 4 * epsilon * (sr6**2 - sr6)

# ── BMH functional forms ──────────────────────────────────────────────────────

def bmh_D(r, B, rho, C, D):
    """Short-range BMH only: B·exp(−r/ρ) − C/r⁶ + D/r⁸"""
    return B * np.exp(-r / rho) - C / r**6 + D / r**8 


def bmh_C(r, B, rho, C):
    """BMH repulsion + dipole-dipole: B·exp(−r/ρ) − C/r⁶"""
    return B * np.exp(-r / rho) - C / r**6


def bmh_rep(r, B, rho):
    """BMH repulsion only: B·exp(−r/ρ)"""
    return B * np.exp(-r / rho)


# ── Fitting ───────────────────────────────────────────────────────────────────

def fit_bmh(sym1, sym2, r_values, e_pot):
    """
    Fit BMH parameters to the reference-subtracted short-range energy.

    Reference subtraction: E_sr(r) = [E_DFT(r) − E_DFT(r_max)] − k·q₁q₂·(1/r − 1/r_max)
    Removes the ionic asymptote offset so E_sr → 0 at r_max.
    Fits pure BMH (no Coulomb) to E_sr; parameters go directly into LAMMPS born/coul/long.

    σ is fixed from Shannon radii; A is back-computed as A = B·exp(−σ/ρ) after fit.

    Returns:
      (params, err) triples for tier1 [A,σ,ρ,C,D], tier2 [A,σ,ρ,C,0],
      tier3 [A,σ,ρ,0,0]  — params in eV, Å, Å, eV·Å⁶, eV·Å⁸
    """
    pair     = (sym1, sym2)
    pair_rev = (sym2, sym1)
    sigma = SIGMA_CONTACT.get(pair, SIGMA_CONTACT.get(pair_rev, SIGMA_DEFAULT))
    q1 = CHARGES.get(sym1, 0.0)
    q2 = CHARGES.get(sym2, 0.0)

    coul_vals = coul(q1, q2, r_values)

    try:
        popt1, pcov1 = curve_fit(
            bmh_D+coul_vals, r_values, e_pot,
            p0     = [500.0, 0.30, 5.0, 2.0],
            bounds = ([0, 0.10, 0, 0], [1e6, 0.60, 500.0, 500.0]),
            maxfev = 100_000,
        )
        B1, rho1, C1, D1 = popt1
        eb1, er1, ec1, ed1 = np.sqrt(np.diag(pcov1))

        popt2, pcov2 = curve_fit(
            bmh_C+coul_vals, r_values, e_pot,
            p0     = [500.0, 0.30, 5.0],
            bounds = ([0, 0.10, 0], [1e6, 0.60, 500.0]),
            maxfev = 100_000,
        )
        B2, rho2, C2 = popt2
        eb2, er2, ec2 = np.sqrt(np.diag(pcov2))

        popt3, pcov3 = curve_fit(
            bmh_rep+coul_vals, r_values, e_pot,
            p0     = [500.0, 0.30],
            bounds = ([0, 0.10], [1e6, 0.60]),
            maxfev = 100_000,
        )
        B3, rho3 = popt3
        eb3, er3 = np.sqrt(np.diag(pcov3))

    except RuntimeError as exc:
        print(f"    WARNING: curve_fit failed: {exc}")
        nan5 = np.full(5, np.nan)
        return (np.zeros(5), nan5), (np.zeros(5), nan5), (np.zeros(5), nan5)

    # Back-compute A from B = A·exp(σ/ρ)  →  A = B·exp(−σ/ρ)
    A1 = B1 * np.exp(-sigma / rho1)
    A2 = B2 * np.exp(-sigma / rho2)
    A3 = B3 * np.exp(-sigma / rho3)

    # Error propagation (σ fixed): δA = sqrt((exp(−σ/ρ)·δB)² + (A·σ/ρ²·δρ)²)
    err_A1 = np.sqrt((np.exp(-sigma / rho1) * eb1)**2 + (A1 * sigma / rho1**2 * er1)**2)
    err_A2 = np.sqrt((np.exp(-sigma / rho2) * eb2)**2 + (A2 * sigma / rho2**2 * er2)**2)
    err_A3 = np.sqrt((np.exp(-sigma / rho3) * eb3)**2 + (A3 * sigma / rho3**2 * er3)**2)

    tier1 = (np.array([A1, sigma, rho1, C1, D1]), np.array([err_A1, 0.0, er1, ec1, ed1]))
    tier2 = (np.array([A2, sigma, rho2, C2, 0.0]), np.array([err_A2, 0.0, er2, ec2, 0.0]))
    tier3 = (np.array([A3, sigma, rho3, 0.0, 0.0]), np.array([err_A3, 0.0, er3, 0.0, 0.0]))
    return tier1, tier2, tier3


def fit_lj(sym1, sym2, r_values, e_pot):
    """
    Fit Lennard-Jones parameters to the reference-subtracted short-range energy.

    Model: V_LJ(r) = 4ε[(σ/r)¹² − (σ/r)⁶]
    Target: E_sr — same reference-subtracted data used by fit_bmh.

    Returns (params, err): params = [epsilon(eV), sigma(Å)]
    """
    idx_min = np.argmin(e_pot)
    sig0 = r_values[idx_min] / 2**(1/6) if e_pot[idx_min] < 0 else 2.0
    eps0 = max(-e_pot[idx_min], 0.05)
    coul_vals = coul(CHARGES.get(sym1, 0.0), CHARGES.get(sym2, 0.0), r_values)
    try:
        popt, pcov = curve_fit(
            lj+coul_vals, r_values, e_pot,
            p0     = [eps0, sig0],
            bounds = ([1e-6, 0.3], [50.0, 6.0]),
            maxfev = 100_000,
        )
        epsilon, sigma_lj = popt
        e_eps, e_sig = np.sqrt(np.diag(pcov))
    except RuntimeError as exc:
        print(f"    WARNING: LJ curve_fit failed: {exc}")
        return np.full(2, np.nan), np.full(2, np.nan)

    return np.array([epsilon, sigma_lj]), np.array([e_eps, e_sig])


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

        e_pot                    = scan_pair(sym1, sym2, r_values)
        tier1, tier2, tier3      = fit_bmh(sym1, sym2, r_values, e_pot)
        params, err              = tier1        # tier1 (full BMH) → LAMMPS output
        A, sigma, rho, C, D     = params

        q1 = CHARGES.get(sym1, 0.0)
        q2 = CHARGES.get(sym2, 0.0)
        r_dense = np.linspace(R_MIN, R_MAX, 300)
        B_model = A * np.exp(sigma / rho)
        e_model = bmh_D(r_dense, B_model, rho, C, D) + coul(q1, q2, r_dense)

        lj_params, lj_err = fit_lj(sym1, sym2, r_values, e_pot)
        eps_lj, sig_lj = lj_params
        # LJ+coul dense curve for plotting against raw e_pot
        e_lj_dense = lj(r_dense, eps_lj, sig_lj) + coul(q1, q2, r_dense)

        results[(sym1, sym2)] = {
            'A': A, 'sigma': sigma, 'rho': rho, 'C': C, 'D': D,
            'err': err,
            'tier2': tier2, 'tier3': tier3,
            'r': r_values, 'e_pot': e_pot,
            'r_dense': r_dense, 'e_model': e_model,
            'lj_params': lj_params, 'lj_err': lj_err, 'e_lj_dense': e_lj_dense,
        }
        print(f"\n  ── Fitted Born-Mayer-Huggins parameters ──────────────────")
        print(f"     A   = {A:10.4f}  ± {err[0]:.4f}   eV")
        print(f"     σ   = {sigma:10.5f}  ± {err[1]:.5f}   Å")
        print(f"     ρ   = {rho:10.5f}  ± {err[2]:.5f}   Å")
        print(f"     C   = {C:10.4f}  ± {err[3]:.4f}   eV·Å⁶")
        print(f"     D   = {D:10.4f}  ± {err[4]:.4f}   eV·Å⁸")
        print(f"\n  ── Fitted Lennard-Jones parameters ───────────────────────")
        print(f"     ε   = {eps_lj:10.5f}  ± {lj_err[0]:.5f}   eV")
        print(f"     σ   = {sig_lj:10.5f}  ± {lj_err[1]:.5f}   Å")
        
        # Evaluate each tier's BMH+Coulomb at r_values
        A1, sig1, rho1, C1, D1 = tier1[0]
        A2, sig2, rho2, C2, _  = tier2[0]
        A3, sig3, rho3, _,  _  = tier3[0]
        e_t1 = bmh_D(r_values, A1 * np.exp(sig1 / rho1), rho1, C1, D1) + coul(q1, q2, r_values)
        e_t2 = bmh_C(r_values, A2 * np.exp(sig2 / rho2), rho2, C2)     + coul(q1, q2, r_values)
        e_t3 = bmh_rep(r_values, A3 * np.exp(sig3 / rho3), rho3)        + coul(q1, q2, r_values)
        e_lj_vals = lj(r_values, eps_lj, sig_lj) + coul(q1, q2, r_values)

        plt.figure(figsize=(6, 4))
        plt.plot(r_values, e_pot,     'o',   color='steelblue', label='DFT E_pot')
        plt.plot(r_values, e_t1,      'r-',  label='BMH+coul rep+C+D')
        plt.plot(r_values, e_t2,      'g--', label='BMH+coul rep+C')
        plt.plot(r_values, e_t3,      'b-.', label='BMH+coul rep only')
        plt.plot(r_values, e_lj_vals, 'm:',  lw=2, label=f'LJ+coul ε={eps_lj:.3f} σ={sig_lj:.3f}')
        plt.axhline(0, color='k', lw=0.8, alpha=0.4)
        plt.xlabel('r (Å)')
        plt.ylabel('Energy (eV)')
        plt.title(f'Fit for {sym1}–{sym2}  (σ={sigma:.3f} Å fixed)')
        plt.legend(fontsize=7)
        plt.show()

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
        fh.write(f"# E_cut: {ECUT_EV} eV   xc: PBE\n")
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

    # ── LAMMPS lj/cut/coul/long pair_coeff table ──────────────────────────────
    print(f"\n{'='*60}")
    print("LAMMPS  pair_coeff  (pair_style lj/cut/coul/long)")
    print(f"{'='*60}")
    print(f"  # {'i':>2} {'j':>2}  {'ε (eV)':>10}  {'σ (Å)':>9}  pair")

    lj_lines = []
    for i, s1 in enumerate(SPECORDER, 1):
        for j, s2 in enumerate(SPECORDER[i-1:], i):
            fwd = (s1, s2)
            rev = (s2, s1)
            dat = results.get(fwd) or results.get(rev)
            if dat and not np.any(np.isnan(dat['lj_params'])):
                eps_lj, sig_lj = dat['lj_params']
                line = (f"pair_coeff {i} {j}  {eps_lj:10.6f}  {sig_lj:.5f}"
                        f"   # {s1}–{s2}")
            else:
                line = (f"pair_coeff {i} {j}      0.000000  1.00000"
                        f"   # {s1}–{s2}  (Coulomb only)")
            print(f"  {line}")
            lj_lines.append(line)

    lj_out = 'LJ_pair_coeff.txt'
    with open(lj_out, 'w') as fh:
        fh.write("# Lennard-Jones pair coefficients\n")
        fh.write("# Fitted from GPAW/PBE dimer energy scans\n")
        fh.write(f"# r scan: {R_MIN:.1f}–{R_MAX:.1f} Å  ({N_R} points)\n")
        fh.write(f"# E_cut: {ECUT_EV} eV   xc: PBE\n")
        fh.write("# pair_style lj/cut/coul/long\n")
        fh.write("# Format: pair_coeff i j  epsilon(eV)  sigma(Å)\n")
        fh.write("#\n")
        for ln in lj_lines:
            fh.write(ln + "\n")
    print(f"\nLJ pair coefficients written → {lj_out}")

    # ── plot ──────────────────────────────────────────────────────────────────
    n  = len(results)
    nc = min(2, n)
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 4*nr), squeeze=False)
    axes_flat = axes.flatten()

    for ax, ((s1, s2), dat) in zip(axes_flat, results.items()):
        A, sigma, rho, C, D = dat['A'], dat['sigma'], dat['rho'], dat['C'], dat['D']
        eps_lj, sig_lj = dat['lj_params']
        ax.scatter(dat['r'], dat['e_pot'], color='steelblue', s=50,
                   zorder=3, label='GPAW E_pot')
        ax.plot(dat['r_dense'], dat['e_model'], 'r-', lw=1.8,
                label=(f"BMH+coul\n"
                       f"A={A:.1f} σ={sigma:.3f} ρ={rho:.3f}\n"
                       f"C={C:.2f} D={D:.2f}"))
        ax.plot(dat['r_dense'], dat['e_lj_dense'], 'm--', lw=1.4,
                label=f"LJ+coul\nε={eps_lj:.4f} σ={sig_lj:.3f}")
        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.set_xlabel('r (Å)')
        ax.set_ylabel('Energy (eV)')
        ax.set_title(f'{s1}–{s2}')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ylo = max(dat['e_pot'].min() - 0.5, -3.0)
        yhi = min(dat['e_pot'].max() + 0.5, 15.0)
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
