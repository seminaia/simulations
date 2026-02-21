from potentials import lj, calc_gpaw
import numpy as np
from MDAnalysis.analysis import distances

class Utilities:
    def __init__(self,
                 initial_positions=None,
                 box_dimensions=None,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.box_dimensions = box_dimensions
        self.initial_positions = initial_positions

    def compute_lj(self):
        """Compute the potential energy by summing up all pair contributions."""
        energy_potential = 0
        for Ni in np.arange(np.sum(self.number_atoms)-1):
            # Read neighbor list
            neighbor_of_i = self.neighbor_lists[Ni]
            # Measure distance
            rij = self.compute_distance(self.atoms_positions[Ni],
                                        self.atoms_positions[neighbor_of_i],
                                        self.box_size)
            # Measure potential using pre-calculated cross coefficients
            sigma_ij = self.sigma_ij_list[Ni]
            epsilon_ij = self.epsilon_ij_list[Ni]
            energy_potential += np.sum(lj(epsilon_ij, sigma_ij, rij))
        return energy_potential
    def compute_gpaw(self):
        """Compute the potential energy using GPAW."""
        energy_potential = 0
        for Ni in np.arange(np.sum(self.number_atoms)-1):
            calculator_params = {
                "convergence": {"density": 1e-4,
                                "eigenstates": 1e-8,
                                "energy": 1e-6,
                                "forces": 1e-4},
                "eigensolver": {"name": "cg",
                                "niter":5},
                "kpts": {"gamma": True,
                        "size": [2, 2, 1]},
                "maxiter": 500,
                "mixer":{"backend": "pulay",
                        "beta": 0.1,
                        "method": "fullspin",
                        "nmaxold": 5,
                        "weight":100},
                "mode": {"ecut": 520,
                        "name": "pw"},
                "nbands":"nao",
                "occupations": {"name": "fermi-dirac",
                                "width": 0.01},
                "setups": {"Ni": ':d, 6.2'},
                "symmetry":"off",
                "txt": "rlx.txt",
                "xc": "PBE"
                }

            energy_potential += np.sum(calc_gpaw(calculator_params))
        return energy_potential
    
    def compute_distance(self,position_i, positions_j, box_size, only_norm = True):
        """
        Measure the distances between two particles.
        # TOFIX: Move as a function instead of a method?
        """
        rij_xyz = np.nan_to_num(np.remainder(position_i - positions_j
                  + box_size[:3]/2.0, box_size[:3]) - box_size[:3]/2.0)
        if only_norm:
            return np.linalg.norm(rij_xyz, axis=1)
        else:
            return np.linalg.norm(rij_xyz, axis=1), rij_xyz
    
    def compute_force(self, return_vector = True):
        if return_vector: # return a N-size vector
            force_vector = np.zeros((np.sum(self.number_atoms),3))
        else: # return a N x N matrix
            force_matrix = np.zeros((np.sum(self.number_atoms),
                                    np.sum(self.number_atoms),3))
        for Ni in np.arange(np.sum(self.number_atoms)-1):
            # Read neighbor list
            neighbor_of_i = self.neighbor_lists[Ni]
            # Measure distance
            rij, rij_xyz = self.compute_distance(self.atoms_positions[Ni],
                                        self.atoms_positions[neighbor_of_i],
                                        self.box_size, only_norm = False)
            # Measure force using information about cross coefficients
            sigma_ij = self.sigma_ij_list[Ni]
            epsilon_ij = self.epsilon_ij_list[Ni]
            fij_xyz = potentials(epsilon_ij, sigma_ij, rij, derivative = True)
            if return_vector:
                # Add the contribution to both Ni and its neighbors
                force_vector[Ni] += np.sum((fij_xyz*rij_xyz.T/rij).T, axis=0)
                force_vector[neighbor_of_i] -= (fij_xyz*rij_xyz.T/rij).T
            else:
                # Add the contribution to the matrix
                force_matrix[Ni][neighbor_of_i] += (fij_xyz*rij_xyz.T/rij).T
        if return_vector:
            return force_vector
        else:
            return force_matrix
    def wrap_in_box(self):
        for dim in np.arange(3):
            out_ids = self.atoms_positions[:, dim] \
                > self.box_boundaries[dim][1]
            self.atoms_positions[:, dim][out_ids] \
                -= np.diff(self.box_boundaries[dim])[0]
            out_ids = self.atoms_positions[:, dim] \
                < self.box_boundaries[dim][0]
            self.atoms_positions[:, dim][out_ids] \
                += np.diff(self.box_boundaries[dim])[0]
