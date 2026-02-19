#!/usr/bin/env python3
"""
Solve the 3×3 linear system for absolute chemical potentials
(raw energies in → abs μ out → rel μ).
Bit-identical to pymatgen hull vertex when the same 3 rows are used.
"""
import re, json, numpy as np, pandas as pd
from pathlib import Path
from doped.chemical_potentials import CompetingPhasesAnalyzer

# ------------------------------------------------------------------
# config
# ------------------------------------------------------------------
CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "defect_dir": ["Interstitial", "Vacancy", "Combined"],
}

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def parse_formula(f: str):
    return {el: (float(n) if n else 1.0)
            for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", str(f).strip())}

def pick_energy_col(df: pd.DataFrame, prefer: str | None = None):
    if prefer and prefer in df.columns:
        return prefer
    for c in ("DFT Energy (eV/fu)", "Raw DFT Energy (eV/fu)", "Energy (eV/fu)", "Energy_form") :
        if c in df.columns:
            return c
    for c in ("DFT Energy (eV/atom)", "Raw DFT Energy (eV/atom)", "Energy (eV/atom)"):
        if c in df.columns:
            return c
    raise ValueError(f"No raw DFT energy column found. Columns: {list(df.columns)}")

def default_ref(el: str) -> str:
    return {"O": "O2", "N": "N2", "H": "H2", "S": "S8", "P": "P4"}.get(el, el)

def setup_dirs(material: str, defect_dir: list[str]):
    base = f"{material}_results"
    paths = {
        "base": base,
        "chempot": f"{base}/chemical_potentials",
        "defects": {cat: f"{base}/{cat}" for cat in defect_dir},
    }
    for p in paths.values():
        if isinstance(p, dict):
            for v in p.values():
                Path(v).mkdir(parents=True, exist_ok=True)
        else:
            Path(p).mkdir(parents=True, exist_ok=True)
    return paths

# ------------------------------------------------------------------
# core routine
# ------------------------------------------------------------------
def export_chempot_limits(
    csv_path: str,
    phases: list[str],
    ref_map: dict[str, str] | None = None,
    formula_col: str = "Formula",
    energy_col_prefer: str | None = None,
    outfile: str = "chemical_potentials.json",
    decimals: int = 5,
):
    
    df = pd.read_csv(csv_path)
    # 1. keep only the lowest-energy row for each composition
    ecol = pick_energy_col(df, energy_col_prefer)
    df[ecol] = pd.to_numeric(df[ecol], errors="coerce")
    df = df.sort_values(ecol).drop_duplicates(subset=[formula_col], keep="first")

    # 2. per-formula energies
    df["N_atoms"] = df[formula_col].map(lambda f: sum(parse_formula(f).values()))
    if "(eV/atom)" in ecol:
        df["E_raw_fu"] = df[ecol] * df["N_atoms"]
    else:
        df["E_raw_fu"] = df[ecol]

    # 3. elements & stoichiometry matrix
    elements = sorted({el for p in phases for el in parse_formula(p)})
    if len(phases) != len(elements):
        raise ValueError(f"Need {len(elements)} phases for {elements}; got {len(phases)}.")
    for el in elements:
        df[el] = df[formula_col].map(lambda f: parse_formula(f).get(el, 0.0))

    # 4. elemental references (μ° per atom)
    ref_map = ref_map or {el: default_ref(el) for el in elements}
    mu0: dict[str, float] = {}
    for el in elements:
        rf = ref_map[el]
        row = df.loc[df[formula_col] == rf].iloc[0]
        mu0[el] = float(row["E_raw_fu"]) / float(row["N_atoms"])

    # 5. build M (stoichiometry) and g (raw energies eV/f.u.)
    M_rows, g_raw = [], []
    for p in phases:
        row = df.loc[df[formula_col] == p].iloc[0]
        comp = parse_formula(p)
        M_rows.append([comp.get(el, 0.0) for el in elements])
        g_raw.append(float(row["E_raw_fu"]))
    M = np.array(M_rows, dtype=float)
    g_raw = np.array(g_raw, dtype=float)

    # 6. solve M · μ_abs = g_raw  →  absolute potentials
    mu_abs = np.linalg.solve(M, g_raw)               # eV/atom
    mu_rel = mu_abs - np.array([mu0[el] for el in elements])  # relative to refs

    # 7. tidy output
    def rnd(x: float) -> float:
        s = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
        return float(s) if s else 0.0

    key = "-".join(phases)
    out = {
        "limits": {key: {el: rnd(mu_abs[i]) for i, el in enumerate(elements)}},
        "elemental_refs": {el: rnd(mu0[el]) for el in elements},
        "limits_wrt_el_refs": {key: {el: rnd(mu_rel[i]) for i, el in enumerate(elements)}},
    }
    Path(outfile).write_text(json.dumps(out, indent=2))
    return out

# ------------------------------------------------------------------
# run
# ------------------------------------------------------------------
if __name__ == "__main__":
    PHASES = ["La2NiO4", "LaNiO3", "O2"]
    paths = setup_dirs(CONFIG["material"], CONFIG["defect_dir"])
    csv = f"{paths['chempot']}/{CONFIG['material']}_formation_energies.csv"
    res = export_chempot_limits(
        csv,
        phases=PHASES,
        ref_map={"La": "La", "Ni": "Ni", "O": "O2"},
        energy_col_prefer="DFT Energy (eV/atom)",
        outfile=f"{paths['chempot']}/chemical_potentials1.json",
    )
    print(json.dumps(res, indent=2))