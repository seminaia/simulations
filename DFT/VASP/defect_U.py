import re
import csv
from pathlib import Path
from doped.corrections import get_kumagai_correction
import logging
import os
from scipy.constants import k as k_B, elementary_charge as e
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.io.vasp.outputs import Outcar
import doped 
from pymatgen.util.io_utils import micro_pyawk
from warnings import filterwarnings
from doped.core import DefectEntry
from doped.analysis import DefectParser
from doped.thermodynamics import DefectThermodynamics
import json
import pandas as pd
from pymatgen.io.vasp.outputs import Vasprun
k_e = k_B / e # eV/K

# --- Logging & style config ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler("defect_analysis.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
plt.rcdefaults()
plt.style.use(f"{doped.__path__[0]}/utils/doped.mplstyle")
plt.switch_backend('Agg')
plt.rcParams.update({'figure.max_open_warning': 100})
filterwarnings("ignore", category=UserWarning)

class OutcarParser:
    def __init__(self, filename):
        self.filename = filename

    def read_lepsilon(self):
        """Read a LEPSILON run.

        TODO: Document the actual variables.
        """
        try:
            search = []

            def dielectric_section_start(results, match):
                results.dielectric_index = -1

            search.append(
                [
                    r"MACROSCOPIC STATIC DIELECTRIC TENSOR \(",
                    None,
                    dielectric_section_start,
                ]
            )

            def dielectric_section_start2(results, match):
                results.dielectric_index = 0

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.dielectric_index == -1,
                    dielectric_section_start2,
                ]
            )

            def dielectric_data(results, match):
                results.dielectric_tensor[results.dielectric_index, :] = np.array(
                    [float(match[i]) for i in range(1, 4)]
                )
                results.dielectric_index += 1

            search.append(
                [
                    r"^ *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *$",
                    lambda results, _line: results.dielectric_index >= 0
                    if results.dielectric_index is not None
                    else None,
                    dielectric_data,
                ]
            )

            def dielectric_section_stop(results, match):
                results.dielectric_index = None

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.dielectric_index >= 1
                    if results.dielectric_index is not None
                    else None,
                    dielectric_section_stop,
                ]
            )

            self.dielectric_index = None
            self.dielectric_tensor = np.zeros((3, 3))

            def piezo_section_start(results, _match):
                results.piezo_index = 0

            search.append(
                [
                    r"PIEZOELECTRIC TENSOR  for field in x, y, z        \(C/m\^2\)",
                    None,
                    piezo_section_start,
                ]
            )

            def piezo_data(results, match):
                results.piezo_tensor[results.piezo_index, :] = np.array([float(match[i]) for i in range(1, 7)])
                results.piezo_index += 1

            search.append(
                [
                    r"^ *[xyz] +([-0-9.Ee+]+) +([-0-9.Ee+]+)"
                    r" +([-0-9.Ee+]+) *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)*$",
                    lambda results, _line: results.piezo_index >= 0 if results.piezo_index is not None else None,
                    piezo_data,
                ]
            )

            def piezo_section_stop(results, _match):
                results.piezo_index = None
            self.born = []

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.piezo_index >= 1 if results.piezo_index is not None else None,
                    piezo_section_stop,
                ]
            )

            self.piezo_index = None
            self.piezo_tensor = np.zeros((3, 6))

            def born_section_start(results, _match):
                results.born_ion = -1

            search.append([r"BORN EFFECTIVE CHARGES ", None, born_section_start])

            def born_ion(results, match):
                results.born_ion = int(match[1]) - 1
                results.born.append(np.zeros((3, 3)))

            search.append(
                [
                    r"ion +([0-9]+)",
                    lambda results, _line: results.born_ion is not None,
                    born_ion,
                ]
            )

            def born_data(results, match):
                results.born[results.born_ion][int(match[1]) - 1, :] = np.array([float(match[i]) for i in range(2, 5)])

            search.append(
                [
                    r"^ *([1-3]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)$",
                    lambda results, _line: results.born_ion >= 0 if results.born_ion is not None else results.born_ion,
                    born_data,
                ]
            )

            def born_section_stop(results, _match):
                results.born_ion = None

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.born_ion >= 1 if results.born_ion is not None else results.born_ion,
                    born_section_stop,
                ]
            )

            self.born_ion = None
            self.born: np.array(self.born,dtype=float) 

            micro_pyawk(self.filename, search, self)

            self.born = np.array(self.born)
            self.dielectric_tensor = self.dielectric_tensor.tolist()
            self.piezo_tensor = self.piezo_tensor.tolist()
            return self.born.copy(), self.dielectric_tensor.copy(), self.piezo_tensor.copy()

        except Exception as exc:
            raise RuntimeError("LEPSILON OUTCAR could not be parsed.") from exc

    def read_lepsilon_ionic(self):
        """Read the ionic component of a LEPSILON run.

        TODO: Document the actual variables.
        """
        try:
            search = []

            def dielectric_section_start(results, _match):
                results.dielectric_ionic_index = -1

            search.append(
                [
                    r"MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC",
                    None,
                    dielectric_section_start,
                ]
            )

            def dielectric_section_start2(results, _match):
                results.dielectric_ionic_index = 0

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.dielectric_ionic_index == -1
                    if results.dielectric_ionic_index is not None
                    else results.dielectric_ionic_index,
                    dielectric_section_start2,
                ]
            )

            def dielectric_data(results, match):
                results.dielectric_ionic_tensor[results.dielectric_ionic_index, :] = np.array(
                    [float(match[i]) for i in range(1, 4)]
                )
                results.dielectric_ionic_index += 1

            search.append(
                [
                    r"^ *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+) *$",
                    lambda results, _line: results.dielectric_ionic_index >= 0
                    if results.dielectric_ionic_index is not None
                    else results.dielectric_ionic_index,
                    dielectric_data,
                ]
            )

            def dielectric_section_stop(results, _match):
                results.dielectric_ionic_index = None

            search.append(
                [
                    r"-------------------------------------",
                    lambda results, _line: results.dielectric_ionic_index >= 1
                    if results.dielectric_ionic_index is not None
                    else results.dielectric_ionic_index,
                    dielectric_section_stop,
                ]
            )

            self.dielectric_ionic_index = None
            self.dielectric_ionic_tensor = np.zeros((3, 3))

            def piezo_section_start(results, _match):
                results.piezo_ionic_index = 0

            search.append(["PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        ", None, piezo_section_start])

            def piezo_data(results, match):
                results.piezo_ionic_tensor[results.piezo_ionic_index, :] = np.array(
                    [float(match[i]) for i in range(1, 7)]
                )
                results.piezo_ionic_index += 1

            search.append(
                [
                    r"^ *[xyz] +([-0-9.Ee+]+) +([-0-9.Ee+]+)"
                    r" +([-0-9.Ee+]+) *([-0-9.Ee+]+) +([-0-9.Ee+]+) +([-0-9.Ee+]+)*$",
                    lambda results, _line: results.piezo_ionic_index >= 0
                    if results.piezo_ionic_index is not None
                    else results.piezo_ionic_index,
                    piezo_data,
                ]
            )

            def piezo_section_stop(results, _match):
                results.piezo_ionic_index = None

            search.append(
                [
                    "-------------------------------------",
                    lambda results, _line: results.piezo_ionic_index >= 1
                    if results.piezo_ionic_index is not None
                    else results.piezo_ionic_index,
                    piezo_section_stop,
                ]
            )

            self.piezo_ionic_index = None
            self.piezo_ionic_tensor = np.zeros((3, 6))

            micro_pyawk(self.filename, search, self)

            self.dielectric_ionic_tensor = self.dielectric_ionic_tensor.tolist()
            self.piezo_ionic_tensor = self.piezo_ionic_tensor.tolist()
            return self.dielectric_ionic_tensor.copy(), self.piezo_ionic_tensor.copy()
        except Exception as exc:
            raise RuntimeError("ionic part of LEPSILON OUTCAR could not be parsed.") from exc

CONFIG = {
    "material": "La2NiO4",
    "elements": ["La", "Ni", "O"],
    "defect_categories": ['Interstitial', 'Vacancy'],
    "defect_dir": ['Interstitial', 'Vacancy', 'Combined'],
    "e_above_hull": 0.09,
    "processes": 4,
}
def validate_dielectric_tensors(outcar_path, output_dir):
    """Extract and save dielectric tensors from OUTCAR"""
    try:
        logger.info(f"Processing dielectric tensors from {outcar_path}")
        outcar = OutcarParser(outcar_path)
        born, static, piezo = outcar.read_lepsilon()
        ionic, piezo_ionic = outcar.read_lepsilon_ionic()
        static = np.asarray(static, dtype=np.float64).reshape((3,3))
        ionic = np.asarray(ionic, dtype=np.float64).reshape((3,3))
        total = np.array(static) + np.array(ionic)
        total = np.asarray(total, dtype=np.float64).reshape((3,3))
        piezo = np.asarray(piezo, dtype=np.float64).reshape((3,6))
        piezo_ionic = np.asarray(piezo_ionic, dtype=np.float64).reshape((3,6))
        born = np.asarray(born, dtype=np.float64).reshape((-1,3,3))
        dielectric_data = {
            "static_dielectric_tensor": static.tolist(),
            "ionic_dielectric_tensor": ionic.tolist(),
            "total_dielectric_tensor": total.tolist(),
            "piezoelectric_tensor": piezo.tolist(),
            "piezoelectric_ionic_tensor": piezo_ionic.tolist(),
            "born_effective_charges": born.tolist()
        }
        # Save dielectric data
        json_path = os.path.join(output_dir, "dielectric_tensors.json")
        with open(json_path, 'w') as f:
            json.dump(dielectric_data["total_dielectric_tensor"], f, indent=4)
        logger.info(f"Saved dielectric tensors to {json_path}")
        return dielectric_data
    except Exception as e:
        logger.error(f"Dielectric extraction failed: {str(e)}", exc_info=True)
        return None  
    
        
def setup_directories(material, defect_dir):
    """Create directory structure and return paths"""
    base_dir = f"{material}_results"
    paths = {
        "base": base_dir,
        "dielectric": os.path.join(base_dir, "dielectric"),
        "chempot": os.path.join(base_dir, "chemical_potentials"),
        "defects": {cat: os.path.join(base_dir, cat) for cat in defect_dir},
    }
    for path in paths.values():
        if isinstance(path, dict):
            for p in path.values():
                os.makedirs(p, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
    return paths
paths = setup_directories(CONFIG['material'], CONFIG['defect_dir'])
U_VALUES = [4,5,6,7,8]
DEFECTS = ["O_i_C2v_0", "O_i_C2v_-1", "O_i_C2v_-2"]

dielectric_path = os.path.join(f"{CONFIG['material']}_defects3", "bulk", "vib","dfpt", "OUTCAR")
dielectric_data = validate_dielectric_tensors(dielectric_path, paths['dielectric'])
print('Dielectric Tensor:', dielectric_data['total_dielectric_tensor'])

TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-\d\.]+)")

def read_toten(outcar):
    with open(f"{outcar}/OUTCAR", "r", errors="ignore") as f:
        for line in reversed(f.readlines()):
            m = TOTEN_RE.search(line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"TOTEN not found in {outcar}")

def charge(name):
    return int(name.split("_")[-1])

rows = []

for U in U_VALUES:
    top_dir = os.path.join(f"{CONFIG['material']}_defects4", "U")
    bulk_path   = os.path.join(top_dir, "bulk", f"U_{U}")
    E_bulk = read_toten(bulk_path)
    bulk_vr = Vasprun(os.path.join(bulk_path, "vasprun.xml"))

    for d in DEFECTS:
        defect_path = os.path.join(top_dir, d, f"U_{U}")
        E_def = read_toten(defect_path)
        q = charge(d)
        dp = DefectParser.from_paths(defect_path,
                                    bulk_path = bulk_path,
                                    dielectric= dielectric_data['total_dielectric_tensor'])
        defect_entry = dp.defect_entry
        thermo = DefectThermodynamics(defect_entries=[defect_entry])        
        corr, fig = get_kumagai_correction(
            defect_entry=defect_entry,
            dielectric= dielectric_data['total_dielectric_tensor'],
            defect_outcar=os.path.join(defect_path, "OUTCAR"),
            bulk_outcar=os.path.join(bulk_path, "OUTCAR"),
            filename = f"kumagai_correction_{d}_U{U}.png",
            plot = True
        )
        E_corr = corr.correction_energy
        print(f"Kumagai correction for {d} (q={q}) at U={U}: {E_corr} eV")
        E_F = bulk_vr.efermi        
        E_form_corr = thermo.get_formation_energy(defect_entry=defect_entry,
                                             fermi_level=0.0)
        E_vbm = dp.bulk_vr.eigenvalue_band_properties[2]
        gap = dp.bulk_vr.eigenvalue_band_properties[0]
        E_form = E_def - E_bulk + q*(E_vbm)
        print(f"VBM: {E_vbm} eV, gap: {gap} eV, E_F: {E_F} eV")
        print(f"Formation energy for {d} (q={q}) at U={U}: Corrected: {E_form_corr} eV, Uncorrected: {E_form} eV)")
        rows.append({
            "defect": d,
            "q": q,
            "U_eV": U,
            "E_bulk_eV": E_bulk,
            "E_defect_eV": E_def,
            "qE_vbm_eV": q*E_vbm,
            "qE_F_eV": q*E_F,
            "E_form_eV": E_form,
            "E_corr_eV": E_corr,
            "E_form_corr_eV": E_form_corr,
        })
with open("formation_energy_U.csv","w",newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("DONE → formation_energy_U.csv")

df = pd.read_csv('formation_energy_U.csv')
plt.figure(figsize=(6,4))

for (defect, q), subdf in df.groupby(["defect", "q"]):
    plt.plot(
        subdf["U_eV"],
        subdf["E_form_corr_eV"],
        marker="o",
        label=f"{defect}, q={q}"
    )

plt.xlabel("U (eV)")
plt.ylabel("Formation energy (eV)")
plt.legend()
plt.tight_layout()
plt.show()


