from itertools import combinations

from reaktoro import (
    NasaDatabase,
    PhreeqcDatabase,
    SupcrtDatabase,
    ThermoFunDatabase,
)


SEARCH_ELEMENTS = ("Sn","Fe","C","O")


DATABASES = [
    (name, ThermoFunDatabase)
    for name in (
        "aq17",
        "cemdata18",
        "heracles",
        "mines16",
        "psinagra-12-07",
        "slop98-organic",
        "slop98",
    )
]
DATABASES += [
    (name, PhreeqcDatabase)
    for name in (
        "Amm.dat",
        "ColdChem.dat",
        "core10.dat",
        "frezchem.dat",
        "iso.dat",
        "llnl.dat",
        "minteq.dat",
        "minteq.v4.dat",
        "phreeqc.dat",
        "pitzer.dat",
        "sit.dat",
        "Tipping_Hurley.dat",
        "wateq4f.dat",
    )
]
DATABASES += [
    (name, SupcrtDatabase)
    for name in ("supcrt98", "supcrt07", "supcrt16", "supcrtbl")
]
DATABASES.append(("nasa-cea", NasaDatabase))


def find_species_by_elements(*elements):
    targets = tuple(
        dict.fromkeys(
            element.strip().capitalize() for element in elements if element.strip()
        )
    )
    if len(targets) < 2:
        raise ValueError("Provide at least two element symbols.")

    pairs = tuple(combinations(targets, 2))
    for database_name, database_type in DATABASES:
        try:
            database = database_type(database_name)
            species_data = [
                (
                    species,
                    {str(symbol) for symbol in species.elements().symbols()},
                )
                for species in database.species()
                if species.name() != "e-" and species.formula().str() != "e-"
            ]
        except RuntimeError as error:
            print(f"{database_name}: COULD NOT LOAD ({error})")
            continue

        print(f"\n{database_name}")
        for pair in pairs:
            pair_symbols = set(pair)
            matches = [
                species
                for species, symbols in species_data
                if pair_symbols.issubset(symbols)
            ]
            print(f"  {' + '.join(pair)}: {len(matches)} species")
            for species in matches:
                print(f"    {species.name()} ({species.formula().str()})")


def find_elements(*elements):
    targets = {
        element.strip().capitalize() for element in elements if element.strip()
    }
    if not targets:
        raise ValueError("Provide at least one element symbol.")

    for database_name, database_type in DATABASES:
        try:
            database = database_type(database_name)
        except RuntimeError as error:
            print(f"{database_name}: COULD NOT LOAD ({error})")
            continue

        print(f"{database_name}:")
        available_elements = sorted(
            database.elements(), key=lambda element: element.symbol()
        )
        matches = [
            element for element in available_elements if element.symbol() in targets
        ]
        for element in matches:
            print(f"  {element.symbol()} ({element.molarMass():.6f} kg/mol)")


if __name__ == "__main__":
    print(f"Elements requested: {' '.join(SEARCH_ELEMENTS)}")
    find_elements(*SEARCH_ELEMENTS)
    print(f"\nSpecies selected by elements: {' '.join(SEARCH_ELEMENTS)}")
    find_species_by_elements(*SEARCH_ELEMENTS)
