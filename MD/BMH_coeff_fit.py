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
from ase.units import eV, Ang, Bohr
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac, Mixer

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

# r grid for dimer scan — expressed as fractions of the pair's contact distance σ
R_FRAC_MIN = 0.50   # lower bound: 50% of σ  (deep repulsive wall)
R_FRAC_MAX = 1.25   # upper bound: 125% of σ (well into flat / zero region)
R_ABS_MIN  = 0.8    # Å absolute floor to avoid DFT core divergence
N_R        = 7     # number of separation points per pair

# GPAW plane-wave settings
ECUT_EV =800   # eV
VACUUM  = 6.0 * Ang   # Å vacuum on each side of the dimer

# Coulomb constant  k_e  in eV·Å  (= e/(4πε₀) in SI, converted to eV·Å)
K_COULOMB = e * 1e10 / (4 * np.pi * epsilon_0)

# Relative dielectric constant ε_r (1 = vacuum; set >1 for screened Coulomb)
# Must match the 'dielectric' command in LAMMPS.
EPSILON_R = 1.0

# ── GPAW helpers ──────────────────────────────────────────────────────────────
def make_gpaw(txt='-'):
    """GPAW PW calculator for isolated dimers (Γ-point, mild smearing)."""
    base_params = {
        "convergence": {"density": 1e-8,
                        "eigenstates": 1e-10,
                        "energy": 1e-6, 
                        "forces": 1e-6},
        "eigensolver": {"name": "rmm-diis",
                        "niter": 5},
        #"hund": True,
        "kpts": (1, 1, 1),
        "maxiter": 1000,
        "mixer": {"backend": "pulay", 
                  "beta": 0.25,
                  "method": "fullspin",
                  "nmaxold": 5,
                  "weight": 50.0},
        "mode": {"name": "pw",
                 "ecut": ECUT_EV},
        "nbands": "nao",
        #"symmetry": "off",
        "occupations": {"name": "methfessel-paxton",
                        "width": 0.01},
        "txt": txt,  
        "xc": {'backend': 'pw',
               'fraction': 0.25,
               'omega': 0.2 * Bohr,  #bohr^-1
               'name': 'HYB_GGA_XC_HSE06'
               },  
    }
    return GPAW(**base_params)


def dimer_atoms(sym1, sym2, r, vacuum=VACUUM):
    """Neutral atom dimer: sym1 at origin, sym2 at (r,0,0), in vacuum box."""
    atoms = Atoms([sym1, sym2], positions=[(0, 0, 0), (r, 0, 0)], pbc=False)
    atoms.center(vacuum=vacuum)
    return atoms


def get_energy(atoms, log_tag):
    """Return potential energy (eV), loading from .gpw checkpoint if available."""
    import os
    gpw_file = f'gpaw_{log_tag}.gpw'
    if os.path.exists(gpw_file) and os.path.getsize(gpw_file) > 0:
        print(f"    Loading from checkpoint: {gpw_file}")
        atoms.calc = GPAW(gpw_file)
        return atoms.get_potential_energy()
    atoms.calc = make_gpaw(txt=f'gpaw_{log_tag}.log')
    energy = atoms.get_potential_energy()
    atoms.calc.write(gpw_file)
    return energy


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

    Reference subtraction: E_sr(r) = [E_DFT(r)  − k·q₁q₂·1/r ]    
    Fits pure BMH (no Coulomb) to E_sr; parameters go directly into LAMMPS born/coul/long.

    σ is fixed from Shannon radii; A is back-computed as A = B·exp(−σ/ρ) after fit.− E_DFT(r_max)

    Returns:
      (params, err) triples for tier1 [A,σ,ρ,C,D], tier2 [A,σ,ρ,C,0],
      tier3 [A,σ,ρ,0,0]  — params in eV, Å, Å, eV·Å⁶, eV·Å⁸
    """
    pair     = (sym1, sym2)
    pair_rev = (sym2, sym1)
    sigma = SIGMA_CONTACT.get(pair, SIGMA_CONTACT.get(pair_rev, SIGMA_DEFAULT))
    q1 = CHARGES.get(sym1, 0.0)
    q2 = CHARGES.get(sym2, 0.0)

    try:
        popt1, pcov1 = curve_fit(
            lambda r, B, rho, C, D: bmh_D(r, B, rho, C, D) + coul(q1, q2, r),
            r_values, e_pot,
            p0     = [500.0, 0.30, 5.0, 2.0],
            bounds = ([0, 0.10, 0, 0], [1e6, 0.60, 500.0, 500.0]),
            maxfev = 100_000,
        )
        B1, rho1, C1, D1 = popt1
        eb1, er1, ec1, ed1 = np.sqrt(np.diag(pcov1))

        popt2, pcov2 = curve_fit(
            lambda r, B, rho, C: bmh_C(r, B, rho, C) + coul(q1, q2, r),
            r_values, e_pot,
            p0     = [500.0, 0.30, 5.0],
            bounds = ([0, 0.10, 0], [1e6, 0.60, 500.0]),
            maxfev = 100_000,
        )
        B2, rho2, C2 = popt2
        eb2, er2, ec2 = np.sqrt(np.diag(pcov2))

        popt3, pcov3 = curve_fit(
            lambda r, B, rho: bmh_rep(r, B, rho) + coul(q1, q2, r),
            r_values, e_pot,
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
    try:
        popt, pcov = curve_fit(
            lambda r, epsilon, sigma: lj(r, epsilon, sigma) + coul(CHARGES.get(sym1, 0.0), CHARGES.get(sym2, 0.0), r),
            r_values, e_pot,
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
    results  = {}

    # ── scan + fit ────────────────────────────────────────────────────────────
    for sym1, sym2 in SCAN_PAIRS:
        pair = (sym1, sym2)
        sigma_pair = SIGMA_CONTACT.get(pair, SIGMA_CONTACT.get((sym2, sym1), SIGMA_DEFAULT))
        r_lo = max(R_FRAC_MIN * sigma_pair, R_ABS_MIN)
        r_hi = R_FRAC_MAX * sigma_pair
        r_values = np.linspace(r_lo, r_hi, N_R)

        print(f"\n{'='*60}")
        print(f"Pair  {sym1}–{sym2}  "
              f"(q₁={CHARGES.get(sym1,0):+.0f}, q₂={CHARGES.get(sym2,0):+.0f})  "
              f"σ={sigma_pair:.3f} Å  r=[{r_lo:.2f}, {r_hi:.2f}] Å, "
              f"Screening:{0.2} Å^-1 {0.2*Bohr:.2f} bohr⁻¹")
        print(f"{'='*60}")

        e_pot                    = scan_pair(sym1, sym2, r_values)
        tier1, tier2, tier3      = fit_bmh(sym1, sym2, r_values, e_pot)
        params1, err1              = tier1        # tier1 (full BMH) → LAMMPS output
        params2, err2              = tier2
        params3, err3              = tier3
        A1, sigma1, rho1, C1, D1   = params1
        err_A1, _, er1, ec1, ed1   = err1
        A2, sigma2, rho2, C2, _    = params2
        err_A2, _, er2, ec2, _     = err2
        A3, sigma3, rho3, _,  _    = params3
        err_A3, _, er3, _, _       = err3
        q1 = CHARGES.get(sym1, 0.0)
        q2 = CHARGES.get(sym2, 0.0)
        r_dense = np.linspace(r_lo, r_hi, 300)
        B1_model = A1 * np.exp(sigma1 / rho1)
        e1_model = bmh_D(r_dense, B1_model, rho1, C1, D1) + coul(q1, q2, r_dense)
        B2_model = A2 * np.exp(sigma2 / rho2)
        e2_model = bmh_C(r_dense, B2_model, rho2, C2) + coul(q1, q2, r_dense)
        B3_model = A3 * np.exp(sigma3 / rho3)
        e3_model = bmh_rep(r_dense, B3_model, rho3) + coul(q1, q2, r_dense)
        lj_params, lj_err = fit_lj(sym1, sym2, r_values, e_pot)
        eps_lj, sig_lj = lj_params
        # LJ+coul dense curve for plotting against raw e_pot
        e_lj_dense = lj(r_dense, eps_lj, sig_lj) + coul(q1, q2, r_dense)
        e_coul = coul(q1, q2, r_dense)
        results[(sym1, sym2)] = {
            'tier1': {'A': A1, 'sigma': sigma1, 'rho': rho1, 'C': C1, 'D': D1, 'err': err_A1, 'err_rho': er1, 'err_C': ec1, 'err_D': ed1},
            'tier2': {'A': A2, 'sigma': sigma2, 'rho': rho2, 'C': C2, 'D': 0.0, 'err': err_A2, 'err_rho': er2, 'err_C': ec2},
            'tier3': {'A': A3, 'sigma': sigma3, 'rho': rho3, 'C': 0.0, 'D': 0.0, 'err': err_A3, 'err_rho': er3},
            'r': r_values, 'e_pot': e_pot,
            'r_dense': r_dense, 'e1_model': e1_model, 'e2_model': e2_model, 'e3_model': e3_model,
            'lj_params': lj_params, 'lj_err': lj_err, 'e_lj_dense': e_lj_dense, 'e_coul': e_coul,
        }
        print(f"     Coulomb only = {e_coul[0]:.5f} eV at r={r_dense[0]:.2f} Å")
        print(f"     BMH+Coulomb (rep+C+D) fit: A={A1:.1f}±{err_A1:.1f} eV, ρ={rho1:.3f}±{er1:.3f} Å, C={C1:.2f}±{ec1:.2f} eV·Å⁶, D={D1:.2f}±{ed1:.2f} eV·Å⁸")
        print(f"     BMH+Coulomb (rep+C) fit: A={A2:.1f}±{err_A2:.1f} eV, ρ={rho2:.3f}±{er2:.3f} Å, C={C2:.2f}±{ec2:.2f} eV·Å⁶")
        print(f"     BMH+Coulomb (rep only) fit: A={A3:.1f}±{err_A3:.1f} eV, ρ={rho3:.3f}±{er3:.3f} Å")
        print(f"     LJ+Coulomb fit: ε={eps_lj:.3f}±{lj_err[0]:.3f} eV, σ={sig_lj:.3f}±{lj_err[1]:.3f} Å")
        print(f"     Coulomb only = {e_coul[0]:.5f} eV at r={r_dense[0]:.2f} Å")
        plt.figure(figsize=(6, 4))
        plt.plot(r_values, e_pot,     'o',   color='steelblue', label='DFT E_pot')
        plt.plot(r_dense, e1_model,  'r-',  label='BMH+coul rep+C+D')
        plt.plot(r_dense, e2_model,  'g--', label='BMH+coul rep+C')
        plt.plot(r_dense, e3_model,  'b-.', label='BMH+coul rep only')
        plt.plot(r_dense, e_lj_dense, 'm:',  lw=2, label=f'LJ+coul ε={eps_lj:.3f} σ={sig_lj:.3f}')
        plt.plot(r_dense, e_coul, 'k--', label='Coulomb only')
        plt.axhline(0, color='k', lw=0.8, alpha=0.4)
        plt.xlabel('r (Å)')
        plt.ylabel('Energy (eV)')
        plt.title(f'Fit for {sym1}–{sym2}')
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
                A   = dat['tier1']['A']
                sig = dat['tier1']['sigma']
                rho = dat['tier1']['rho']
                C   = dat['tier1']['C']
                D   = dat['tier1']['D']
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
        fh.write(f"# r scan: {R_FRAC_MIN:.0%}–{R_FRAC_MAX:.0%} of σ  ({N_R} points per pair)\n")
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
                A, sigma, rho, C, D = dat['tier1']['A'], dat['tier1']['sigma'], dat['tier1']['rho'], dat['tier1']['C'], dat['tier1']['D']
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
        fh.write(f"# r scan: {R_FRAC_MIN:.0%}–{R_FRAC_MAX:.0%} of σ  ({N_R} points per pair)\n")
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
