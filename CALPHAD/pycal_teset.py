import matplotlib.pyplot as plt
from pycalphad import Database, ternplot, binplot
from pycalphad.core.utils import filter_phases
import pycalphad.variables as v

# Load database
f = "mmc1.TDB"
with open(f, 'r',encoding='latin-1') as file:
    content = file.read()
dbf = Database(content)

bin_comp = ['FE','C','VA']
print("Phases:", phase_keys)

tern_comp = ['FE','SN','MN','VA']
phase_keys = dbf.phases.keys()
print("Phases:", phase_keys)
print("Number of phases:", len(phase_keys))

dof1 = len(bin_comp) - len(phase_keys) + 2
dof2 = len(tern_comp) - len(phase_keys) + 2
print("Degrees of freedom for Binary:", dof1)
print("Degrees of freedom for Ternary:", dof2)

conds_tern = {
    v.T: 1323,
    v.P: 101325,
    v.N: 1,
    v.X('FE'): (0, 1, 0.01),  # vary from 0 to 1 step 0.01
    v.X('SN'): (0, 1, 0.01),  # vary from 0 to 1 step 0.01
}
ax_tern = ternplot(
    dbf,
    tern_comp,
    phase_keys,          # use all active phases, not just 'LIQUID'
    conds_tern,
    x=v.X('FE'),
    y=v.X('SN'),
)

# Conditions for the isothermal section
conds_bin = {
    v.X('C'):  (0, 1, 0.01),
    v.T: (300, 1800, 10),                   
    v.P: 101325,
    v.N: 1
}
 
ax_bin = binplot(
    dbf,
    bin_comp,
    phase_keys,          
    conds_bin,   
)
# Format the plot
fig_tern = ax_tern.figure
ax_tern.set_title("Fe-Sn-Mn Ternary Phase Diagram at 1323 K and 1 atm")
fig_tern.set_size_inches(9, 6)
fig_tern.set_dpi(150)

fig_bin = ax_bin.figure
ax_bin.set_title("Fe-C Binary Phase Diagram at 1 atm and 300-2000 K")
fig_bin.set_size_inches(9, 6)
fig_bin.set_dpi(150)
plt.show()