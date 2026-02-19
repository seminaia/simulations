import os
import numpy as np
from pymatgen.io.vasp import Vasprun
from pymatgen.analysis.elasticity.strain import Deformation
from amset.electronic_structure.kpoints import get_kpoints_from_bandstructure,get_mesh_from_kpoint_diff , get_kpoint_indices, kpoints_to_first_bz
from amset.electronic_structure.symmetry import expand_bandstructure, get_symmops
import logging

logger = logging.getLogger(__name__)
_mapping_tol = 0.002

def get_mesh_from_band_structure(bandstructure):
    kpoints = np.array([k.frac_coords for k in bandstructure.kpoints])
    mesh, is_shifted = get_mesh_from_kpoint_diff(kpoints)
    return tuple(mesh.round().astype(int)), is_shifted
def calculate_deformation(bulk_structure, deformed_structure):
    """Calculate the deformation matrix from bulk to deformed structure."""
    bulk_lattice = bulk_structure.lattice.matrix
    deformed_lattice = deformed_structure.lattice.matrix
    return np.transpose(np.dot(np.linalg.inv(bulk_lattice), deformed_lattice)).round(10)

def get_strain_mapping(bulk_structure, deformation_calculations):
    """Map strains to their corresponding calculations."""
    strain_mapping = {}
    for calc in deformation_calculations:
        deformed_structure = calc["bandstructure"].structure
        matrix = calculate_deformation(bulk_structure, deformed_structure)
        strain = Deformation(matrix).green_lagrange_strain
        strain_mapping[tuple(map(tuple, strain))] = calc
    return strain_mapping

def get_symmetrized_strain_mapping(bulk_structure, strain_mapping, symprec=1e-1):
    """Symmetrize the strain mapping with adjusted symmetry precision."""
    frac_ops = get_symmops(bulk_structure)
    for strain, calc in strain_mapping.items():
        calc["bandstructure"] = expand_bandstructure(calc["bandstructure"], symprec=symprec / 100)
    for strain, calc in list(strain_mapping.items()):
        k_old = get_kpoints_from_bandstructure(calc["bandstructure"], sort=True)
        k_old = kpoints_to_first_bz(k_old)
        for frac_op in frac_ops:
            r_cart = np.dot(bulk_structure.lattice.matrix.T, np.dot(frac_op.rotation_matrix.T, np.linalg.inv(bulk_structure.lattice.matrix.T)))
            tstrain = Deformation(np.dot(strain, r_cart)).green_lagrange_strain
            if tuple(map(tuple, tstrain)) not in strain_mapping:
                rband = expand_bandstructure(calc["bandstructure"], symprec=symprec / 100)
                k_new = get_kpoints_from_bandstructure(rband, sort=True)
                k_new = kpoints_to_first_bz(k_new)
                if np.max(np.linalg.norm(k_old - k_new, axis=1)) < 0.001:
                    tcalc = {"reference": calc["reference"], "bandstructure": rband}
                    strain_mapping[tuple(map(tuple, tstrain))] = tcalc
    return strain_mapping

def get_strain_deformation_potential(strain, bulk_bandstructure, deformation_bandstructure, bulk_reference, deformation_reference):
    """Calculate deformation potential from energy differences."""
    strain = np.array(strain).round(5)
    flat_strain = strain.ravel()
    strain_amount = flat_strain[np.abs(flat_strain).argmax()]
    ref_diff = bulk_reference - deformation_reference
    kpoints = get_kpoints_from_bandstructure(bulk_bandstructure)
    mesh, is_shifted = get_mesh_from_band_structure(bulk_bandstructure)
    indices_to_keep = get_kpoint_indices(kpoints, mesh, is_shifted=is_shifted)
    deform_kpoints = get_kpoints_from_bandstructure(deformation_bandstructure)
    deform_indices = get_kpoint_indices(deform_kpoints, mesh, is_shifted=is_shifted)
    if not set(indices_to_keep).issubset(set(deform_indices)):
        raise RuntimeError("Deformation band structure doesn't contain the same k-points as the bulk.")
    deform_map = np.full(np.max(deform_indices) + 1, -1)
    deform_map[deform_indices] = np.arange(len(deform_indices))
    select_indices = deform_map[indices_to_keep]
    energy_diff = {}
    for spin, spin_origin in bulk_bandstructure.bands.items():
        diff = spin_origin - deformation_bandstructure.bands[spin][:, select_indices]
        diff -= ref_diff
        energy_diff[spin] = np.abs(diff / strain_amount)
    return energy_diff

def calculate_deformation_potentials(bulk_calculation, strain_mapping):
    """Compute deformation potentials for all spins and bands."""
    deformation_potentials = {s: np.zeros(b.shape + (3, 3)) for s, b in bulk_calculation["bandstructure"].bands.items()}
    norm = np.zeros((3, 3))
    for strain, deformation_calculation in strain_mapping.items():
        deform = get_strain_deformation_potential(
            strain, bulk_calculation["bandstructure"], deformation_calculation["bandstructure"],
            bulk_calculation["reference"], deformation_calculation["reference"]
        )
        max_strain = np.abs(strain).max()
        strain_loc = np.abs(strain) > 0.25 * max_strain
        loc_x, loc_y = np.where(strain_loc)
        norm += strain_loc
        for spin, spin_deform in deform.items():
            deformation_potentials[spin][:, :, loc_x, loc_y] += spin_deform[..., None]
    for spin in deformation_potentials:
        deformation_potentials[spin] /= norm[None, None]
    return deformation_potentials

def main():
    base_dir = "La2NiO4_bulk/phonon"
    undisp_path = os.path.join(base_dir, "undisp", "vasprun.xml")
    disp_dirs = [os.path.join(base_dir, f"disp-0{i:02d}", "vasprun.xml") for i in range(1, 13)]

    # Load bulk calculation
    print("Loading undeformed structure...")
    bulk_vr = Vasprun(undisp_path)
    bulk_bs = bulk_vr.get_band_structure()
    bulk_calc = {"bandstructure": bulk_bs, "reference": bulk_vr.final_energy}

    # Load deformed calculations
    deformation_calculations = []
    for disp_path in disp_dirs:
        if os.path.exists(disp_path):
            print(f"Loading {disp_path}...")
            disp_vr = Vasprun(disp_path)
            disp_bs = disp_vr.get_band_structure()
            deformation_calculations.append({"bandstructure": disp_bs, "reference": disp_vr.final_energy})
        else:
            print(f"Warning: {disp_path} not found, skipping.")

    # Get strain mapping
    bulk_structure = bulk_vr.final_structure
    strain_mapping = get_strain_mapping(bulk_structure, deformation_calculations)

    # Symmetrize strain mapping with adjusted tolerance
    symprec = 1e-1  # Increased tolerance to handle numerical noise
    sym_strain_mapping = get_symmetrized_strain_mapping(bulk_structure, strain_mapping, symprec=symprec)

    # Calculate deformation potentials
    deformation_potentials = calculate_deformation_potentials(bulk_calc, sym_strain_mapping)

    # Print and save results
    for spin, potentials in deformation_potentials.items():
        print(f"Deformation potentials for spin {spin}:")
        for band_idx in range(potentials.shape[0]):
            print(f"Band {band_idx}: {potentials[band_idx, 0, 0, 0]:.3f} eV")
    if deformation_potentials:
        np.savez(os.path.join(base_dir, "deformation_potentials.npz"), **{str(spin): pot for spin, pot in deformation_potentials.items()})
        print("Deformation potentials saved to deformation_potentials.npz")

if __name__ == "__main__":
    main() 
