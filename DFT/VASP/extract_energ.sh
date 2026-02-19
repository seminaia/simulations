#!/bin/bash

# Create or overwrite the output CSV files with headers
echo "Directory,Volume (A^3/atom),Energy per atom (eV/atom)" > energy_vol_isif2.csv
echo "Directory,Volume (A^3/atom),Energy per atom (eV/atom)" > energy_vol_isif4.csv

# Loop over each directory matching the pattern lattice_*
for dir in lattice_*; do
    # Initialize number of atoms
    num_atoms="N/A"

    # Get number of atoms from OUTCAR
    if [[ -f "$dir/OUTCAR_STATIC_ISIF_2" ]]; then
        num_atoms=$(grep -m 1 "NIONS = " "$dir/OUTCAR_STATIC_ISIF_2" | awk '{print $12}')
    elif [[ -f "$dir/OUTCAR" ]]; then
        num_atoms=$(grep -m 1 "NIONS = " "$dir/OUTCAR" | awk '{print $12}')
    fi

    # Check if either OUTCAR_RELAX_ISIF2 or OUTCAR_STATIC_ISIF2 exists
    if [[ -f "$dir/OUTCAR_STATIC_ISIF_2" ]]; then
        # Extract volume and total energy from ISIF2 static calculation
        volume_isif2=$(grep "volume of cell" "$dir/OUTCAR_STATIC_ISIF_2" | awk '{print $5}' | tail -1)
        total_energy_isif2=$(grep "TOTEN" "$dir/OUTCAR_STATIC_ISIF_2" | tail -1 | awk '{print $5}')
    elif [[ -f "$dir/OUTCAR_RELAX_ISIF_2" ]]; then
        # Extract volume and total energy from ISIF2 relaxation
        volume_isif2=$(grep "volume of cell" "$dir/OUTCAR_RELAX_ISIF_2" | awk '{print $5}' | tail -1)
        total_energy_isif2=$(grep "TOTEN" "$dir/OUTCAR_RELAX_ISIF_2" | tail -1 | awk '{print $5}')
    else
        volume_isif2="N/A"
        total_energy_isif2="N/A"
    fi

    # Extract volume and total energy from the main OUTCAR (ISIF4 or other calculation)
    if [[ -f "$dir/OUTCAR" ]]; then
        volume_isif4=$(grep "volume of cell" "$dir/OUTCAR" | awk '{print $5}' | tail -1)
        total_energy_isif4=$(grep "TOTEN" "$dir/OUTCAR" | tail -1 | awk '{print $5}')
    else
        volume_isif4="N/A"
        total_energy_isif4="N/A"
    fi

    # Calculate volume per atom and energy per atom, handling cases where num_atoms is not available
    if [[ "$num_atoms" != "N/A" && "$num_atoms" -ne 0 ]]; then
        if [[ "$volume_isif2" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            volume_per_atom_isif2=$(echo "scale=3; $volume_isif2 / $num_atoms" | bc )
        else
            volume_per_atom_isif2="N/A"
        fi
        if [[ "$total_energy_isif2" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
            energy_per_atom_isif2=$(echo "scale=3; $total_energy_isif2 / $num_atoms" | bc)
        else
            energy_per_atom_isif2="N/A"
        fi

        if [[ "$volume_isif4" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            volume_per_atom_isif4=$(echo "scale=3;$volume_isif4 / $num_atoms" | bc )
        else
            volume_per_atom_isif4="N/A"
        fi
        if [[ "$total_energy_isif4" =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
            energy_per_atom_isif4=$(echo "scale=3; $total_energy_isif4 / $num_atoms" | bc)
        else
            energy_per_atom_isif4="N/A"
        fi
    else
        volume_per_atom_isif2="N/A"
        energy_per_atom_isif2="N/A"
        volume_per_atom_isif4="N/A"
        energy_per_atom_isif4="N/A"
    fi

    # Append to CSV for ISIF2 and ISIF4 calculations
    echo "$dir,$volume_per_atom_isif2,$energy_per_atom_isif2" >> energy_vol_isif2.csv
    echo "$dir,$volume_per_atom_isif4,$energy_per_atom_isif4" >> energy_vol_isif4.csv
done

# Add error handling if needed
if [[ $? -ne 0 ]]; then
    echo "Error occurred during script execution." >&2
    exit 1
fi
