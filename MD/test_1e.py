from MinimizeEnergy import MinimizeEnergy
from pint import UnitRegistry
ureg = UnitRegistry()

# Define atom number of each group
nmb_1, nmb_2= [2, 3]
# Define LJ parameters (sigma)
sig_1, sig_2 = [3, 4]*ureg.angstrom
# Define LJ parameters (epsilon)
eps_1, eps_2 = [0.2, 0.4]*ureg.kcal/ureg.mol
# Define atom mass
mss_1, mss_2 = [10, 20]*ureg.gram/ureg.mol
# Define box size
L = 20*ureg.angstrom
# Define a cut off
rc = 2.5*sig_1

# Initialize the prepare object
minimizer = MinimizeEnergy(
    ureg = ureg,
    maximum_steps=100,
    number_atoms=[nmb_1, nmb_2],
    epsilon=[eps_1, eps_2], # kcal/mol
    sigma=[sig_1, sig_2], # A
    atom_mass=[mss_1, mss_2], # g/mol
    box_dimensions=[L, L, L], # A
    cut_off=rc,
)
minimizer.run()

# Test function using pytest
def test_energy_and_force():
    Final_Epot = minimizer.Epot
    Final_MaxF = minimizer.MaxF
    assert Final_Epot < 0, f"Test failed: Final energy too large: {Final_Epot}"
    assert Final_MaxF < 10, f"Test failed: Final max force too large: {Final_MaxF}"
    print("Test passed")

# If the script is run directly, execute the tests
if __name__ == "__main__":
    import pytest
    # Run pytest programmatically
    pytest.main(["-s", __file__])