#!/bin/bash

# Create or overwrite the output CSV files with headers
echo "Directory,Volume (A^3/atom),Ion,Magnetization (microB)" > mag_vol_isif2.csv
echo "Directory,Volume (A^3/atom),Ion,Magnetization (microB)" > mag_vol_isif4.csv

# Loop over each directory matching the pattern lattice_*
for dir in lattice_*; do
    
   num_atoms=$(sed -n '7p' "$dir/POSCAR" | awk '{for(i=1;i<=NF;i++) sum+=$i; print sum}')
    # Check if OUTCAR_STATIC_ISIF_2 exists
    if [[ -f "$dir/OUTCAR_STATIC_ISIF_2" ]]; then

        # Extract volume from ISIF2 static calculation
        volume_isif2=$(grep "volume of cell" "$dir/OUTCAR_STATIC_ISIF_2" | awk '{print $5}' | tail -1)
        volume_per_atom_isif2=$(echo "scale=3;$volume_isif2 / $num_atoms" | bc)

        # Use an associative array to capture the last total magnetization for each ion
        declare -A last_mag_array
        
        # Capture magnetization blocks
        magnetization_data=$(grep -A 20 "magnetization (x)" "$dir/OUTCAR_STATIC_ISIF_2" | grep -v "tot" | grep -v "^\-\-")

        # Process magnetization data to find the last total values
        while read -r line; do
            # Check for the total magnetization line at the end of the block
            if [[ $line =~ ^[[:space:]]*([0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+) ]]; then
                ion_num="${BASH_REMATCH[1]}"
                total_mag="${BASH_REMATCH[6]}"
                last_mag_array["$ion_num"]="$total_mag"  # Store the total magnetization for the ion
            fi
        done <<< "$magnetization_data"

        # Output the last total magnetization values for each ion
        for ion_num in "${!last_mag_array[@]}"; do
            echo "$dir,$volume_per_atom_isif2,Ion $ion_num,${last_mag_array[$ion_num]}" >> mag_vol_isif2.csv
        done

    elif [[ -f "$dir/OUTCAR_RELAX_ISIF_2" ]]; then
        # Extract volume from ISIF2 relaxation
        volume_isif2=$(grep "volume of cell" "$dir/OUTCAR_RELAX_ISIF_2" | awk '{print $5}' | tail -1)
	volume_per_atom_isif2=$(echo "scale=3;$volume_isif2 / $num_atoms" | bc)
        # Use an associative array to capture the last total magnetization for each ion
        declare -A last_mag_array
        
        # Capture magnetization blocks
        magnetization_data=$(grep -A 20 "magnetization (x)" "$dir/OUTCAR_RELAX_ISIF_2" | grep -v "tot" | grep -v "^\-\-")

        # Process magnetization data to find the last total values
        while read -r line; do
            # Check for the total magnetization line at the end of the block
            if [[ $line =~ ^[[:space:]]*([0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+) ]]; then
                ion_num="${BASH_REMATCH[1]}"
                total_mag="${BASH_REMATCH[6]}"
                last_mag_array["$ion_num"]="$total_mag"  # Store the total magnetization for the ion
            fi
        done <<< "$magnetization_data"

        # Output the last total magnetization values for each ion
        for ion_num in "${!last_mag_array[@]}"; do
            echo "$dir,$volume_per_atom_isif2,Ion $ion_num,${last_mag_array[$ion_num]}" >> mag_vol_isif2.csv
        done
    else
        echo "No ISIF2 OUTCAR found in $dir" # Debugging info
        echo "$dir,N/A,N/A,N/A" >> mag_vol_isif2.csv
    fi

    # Extract volume and magnetization for ISIF4 or other calculations in OUTCAR
    if [[ -f "$dir/OUTCAR" ]]; then

        volume_isif4=$(grep "volume of cell" "$dir/OUTCAR" | awk '{print $5}' | tail -1)
	volume_per_atom_isif4=$(echo "scale=3; $volume_isif4 / $num_atoms" | bc)

        # Use an associative array to capture the last total magnetization for each ion
        declare -A last_mag_array

        # Capture magnetization blocks
        magnetization_data=$(grep -A 20 "magnetization (x)" "$dir/OUTCAR" | grep -v "tot" | grep -v "^\-")

        # Process magnetization data to find the last total values
        while read -r line; do
            # Check for the total magnetization line at the end of the block
            if [[ $line =~ ^[[:space:]]*([0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+)[[:space:]]+([-+]?[0-9]*\.?[0-9]+) ]]; then
                ion_num="${BASH_REMATCH[1]}"
                total_mag="${BASH_REMATCH[6]}"
                last_mag_array["$ion_num"]="$total_mag"  # Store the total magnetization for the ion
            fi
        done <<< "$magnetization_data"

        # Output the last total magnetization values for each ion
        for ion_num in "${!last_mag_array[@]}"; do
            echo "$dir,$volume_per_atom_isif4,Ion $ion_num,${last_mag_array[$ion_num]}" >> mag_vol_isif4.csv
        done
    else
        echo "$dir,N/A,N/A,N/A" >> mag_vol_isif4.csv
    fi
done

