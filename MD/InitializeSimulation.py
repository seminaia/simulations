import os
import numpy as np
from Prepare import Prepare
from Utilities import Utilities
from MDAnalysis.analysis import distances


class InitializeSimulation(Prepare, Utilities):
    def __init__(self,
                box_dimensions,  # List - Angstroms
                cut_off, # Angstroms
                initial_positions=None,  # Array - Angstroms
                neighbor=1, # Integer
                thermo_period = None,
                dumping_period = None,
                thermo_outputs = None,
                data_folder="Outputs/",
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)
        self.box_dimensions = box_dimensions
        self.cut_off = cut_off
        self.neighbor = neighbor
        self.step = 0 # initialize simulation step
        self.initial_positions = initial_positions
        self.thermo_period = thermo_period
        self.dumping_period = dumping_period
        self.thermo_outputs = thermo_outputs
        self.data_folder = data_folder
        if os.path.exists(self.data_folder) is False:
            os.mkdir(self.data_folder)
        self.nondimensionalize_units(["box_dimensions", "cut_off", "initial_positions"])
        self.define_box()
        self.populate_box()
        self.update_neighbor_lists()       
        self.update_cross_coefficients()
            
    def nondimensionalize_units(self, quantities_to_normalise):
        for name in quantities_to_normalise:
            quantity = getattr(self, name)  # Get the attribute by name
            if isinstance(quantity, list):
                for i, element in enumerate(quantity):
                    assert element.units in self.ref_units, \
                        f"Error: Units not part of the reference units"
                    ref_value = self.ref_quantities[self.ref_units.index(element.units)]
                    quantity[i] = element/ref_value
                    assert quantity[i].units == self.ureg.dimensionless, \
                        f"Error: Quantities are not properly nondimensionalized"
                    quantity[i] = quantity[i].magnitude # get rid of ureg
                setattr(self, name, quantity)
            elif len(np.shape(quantity)) > 0: # for position array
                assert element.units in self.ref_units, \
                    f"Error: Units not part of the reference units"
                ref_value = self.ref_quantities[self.ref_units.index(element.units)]
                quantity = quantity/ref_value
                assert quantity.units == self.ureg.dimensionless, \
                    f"Error: Quantities are not properly nondimensionalized"
                quantity = quantity.magnitude # get rid of ureg
                setattr(self, name, quantity)
            else:
                if quantity is not None:
                    assert np.shape(quantity) == (), \
                        f"Error: The quantity is a list or an array"
                    assert quantity.units in self.ref_units, \
                        f"Error: Units not part of the reference units"
                    ref_value = self.ref_quantities[self.ref_units.index(quantity.units)]
                    quantity = quantity/ref_value
                    assert quantity.units == self.ureg.dimensionless, \
                        f"Error: Quantities are not properly nondimensionalized"
                    quantity = quantity.magnitude # get rid of ureg
                    setattr(self, name, quantity)
    def define_box(self):
        """Define the simulation box. Only 3D boxes are supported."""
        box_boundaries = np.zeros((3, 2))
        for dim, L in enumerate(self.box_dimensions):
            box_boundaries[dim] = -L/2, L/2
        self.box_boundaries = box_boundaries
        box_size = np.diff(self.box_boundaries).reshape(3)
        box_geometry = np.array([90, 90, 90])
        self.box_size = np.array(box_size.tolist()+box_geometry.tolist())

    def populate_box(self):
        Nat = np.sum(self.number_atoms) # total number of atoms
        if self.initial_positions is None:
            atoms_positions = np.zeros((Nat, 3))
            for dim in np.arange(3):
                diff_box = np.diff(self.box_boundaries[dim])
                random_pos = np.random.random(Nat)
                atoms_positions[:, dim] = random_pos*diff_box-diff_box/2
            self.atoms_positions = atoms_positions
        else:
            self.atoms_positions = self.initial_positions
                        
    def update_neighbor_lists(self, force_update=False):
        if (self.step % self.neighbor == 0) | force_update:
            matrix = distances.contact_matrix(self.atoms_positions,
                cutoff=self.cut_off, #+2,
                returntype="numpy",
                box=self.box_size)
            neighbor_lists = []
            for cpt, array in enumerate(matrix[:-1]):
                list = np.where(array)[0].tolist()
                list = [ele for ele in list if ele > cpt]
                neighbor_lists.append(list)
            self.neighbor_lists = neighbor_lists

    def update_cross_coefficients(self, force_update=False):
        if (self.step % self.neighbor == 0) | force_update:
            # Precalculte LJ cross-coefficients
            sigma_ij_list = []
            epsilon_ij_list = []
            for Ni in np.arange(np.sum(self.number_atoms)-1): # tofix error for GCMC
                # Read information about atom i
                sigma_i = self.atoms_sigma[Ni]
                epsilon_i = self.atoms_epsilon[Ni]
                neighbor_of_i = self.neighbor_lists[Ni]
                # Read information about neighbors j
                sigma_j = self.atoms_sigma[neighbor_of_i]
                epsilon_j = self.atoms_epsilon[neighbor_of_i]
                # Calculare cross parameters
                sigma_ij_list.append((sigma_i+sigma_j)/2)
                epsilon_ij_list.append((epsilon_i+epsilon_j)/2)
            self.sigma_ij_list = sigma_ij_list
            self.epsilon_ij_list = epsilon_ij_list
