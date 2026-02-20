from MonteCarlo import MonteCarlo
from pint import UnitRegistry
ureg = UnitRegistry()
import os

# Define atom number of each group
nmb_1 = 50
nmb_2 = 50  # New group for testing swaps
# Define LJ parameters (sigma)
sig_1 = 3 * ureg.angstrom
sig_2 = 4 * ureg.angstrom  # Different sigma for group 2
# Define LJ parameters (epsilon)
eps_1 = 0.1 * ureg.kcal / ureg.mol
eps_2 = 0.15 * ureg.kcal / ureg.mol  # Different epsilon for group 2
# Define atom mass
mss_1 = 10 * ureg.gram / ureg.mol
mss_2 = 12 * ureg.gram / ureg.mol  # Different mass for group 2
# Define box size
L = 20 * ureg.angstrom
# Define a cut off
rc = 2.5 * sig_1
# Pick the desired temperature
T = 300 * ureg.kelvin

# Initialize the prepare object
mc = MonteCarlo(
    ureg=ureg,
    maximum_steps=100,
    thermo_period=10,
    dumping_period=10,
    number_atoms=[nmb_1, nmb_2],  # Include two groups of atoms for swap
    epsilon=[eps_1, eps_2],  # kcal/mol
    sigma=[sig_1, sig_2],  # A
    atom_mass=[mss_1, mss_2],  # g/mol
    box_dimensions=[L, L, L],  # A
    cut_off=rc,
    thermo_outputs="Epot-press",
    desired_temperature=T,  # K
    neighbor=1,
    swap_type=[0, 1]  # Enable Monte Carlo swap between groups 1 and 2
)

# Run the Monte Carlo simulation
mc.run()

# Test function using pytest
def test_output_files():
    assert os.path.exists("Outputs/dump.mc.lammpstrj"), \
        "Test failed: dump file was not created"
    assert os.path.exists("Outputs/simulation.log"), \
        "Test failed: log file was not created"
    print("Test passed")

# Test the swap counters
def test_swap_counters():
    assert mc.successful_swap + mc.failed_swap > 0, \
        "Test failed: No swaps were attempted"
    print("Swap test passed")

# If the script is run directly, execute the tests
if __name__ == "__main__":
    import pytest
    # Run pytest programmatically
    pytest.main(["-s", __file__])