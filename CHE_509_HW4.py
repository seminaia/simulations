import sympy as sp

# Species: A=N2O5, B=N2O4, C=NO3, D=NO2, E=NO, F=O2
species = ["A", "B", "C", "D", "E", "F"]

rxn = [
    ("OR1", {"A": -2, "B": 2, "F": 1}),
    ("OR2", {"A": -1, "C": 1, "D": 1}),
    ("OR3", {"B": -1, "D": 2}),
    ("OR4", {"C": -1, "E": 1, "F": 1}),
    ("OR5", {"E": -1, "A": -1, "D": 3}),
]

reaction_names = [r[0] for r in rxn]

# Stoichiometric coefficient matrix nu: rows = species, cols = reactions
nu = sp.zeros( len(rxn), len(species))
for i, (name, coeffs) in enumerate(rxn):
    for s, c in coeffs.items():
        nu[i, species.index(s)] = c

print("nu =")
sp.pprint(nu)

R = nu.rank()
print("\nR = rank(nu) =", R)

# Atom matrix M: rows = species, columns = N, O
M = sp.Matrix([
    [2, 5],  # A = N2O5
    [2, 4],  # B = N2O4
    [1, 3],  # C = NO3
    [1, 2],  # D = NO2
    [1, 1],  # E = NO
    [0, 2],  # F = O2
])

print("\nM =")
sp.pprint(M)

m = M.rank()
n = len(species)

print("\nn =", n)
print("m = rank(M) =", m)
print("R = n - m ?", R == n - m)
print("n - m =", n - m)

# Choose OR1, OR2, OR3, OR4 as independent set
ind = [0, 1, 2, 3]  # columns for OR1-OR4
nu_ind = nu.extract(ind, list(range(len(species))))

print("\nnu_ind =")
sp.pprint(nu_ind)

R_ind = nu_ind.rank()
print("\nrank(nu_ind) =", R_ind)

# Confirm independence with a nonzero 4x4 minor
minor = nu.extract([0, 1, 2, 4], [0, 1, 2, 3])
print("\nminor =")
sp.pprint(minor)
print("det(minor) =", minor.det())

# Check dependency: OR1 - OR2 + 2*OR3 - OR4 - OR5 = 0
dep = nu[0, :] - nu[1, :] + 2*nu[2, :] - nu[3, :] - nu[4, :]
print("\nOR1 - OR2 + 2*OR3 - OR4 - OR5 =")
sp.pprint(dep)