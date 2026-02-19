import os
import re
from pymatgen.io.vasp import outputs

# Get the script directory and change to it
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

try:
    # Read the contents of the INCAR file
    with open("INCAR", "r") as f:
        incar_content = f.readlines()

    # Track whether the parameters were replaced
    magmom_replaced = False
    lorbit_replaced = False
    ispin_replaced = False

    # Parse the OUTCAR file to extract magnetization data
    OUTCAR = outputs.Outcar("OUTCAR")
    mag_str = "MAGMOM = " + " ".join([str(m["tot"]) for m in OUTCAR.magnetization])

    # Prepare to overwrite the INCAR file
    with open("INCAR", "w") as f:
        for line in incar_content:
            # Use regex to replace any line containing MAGMOM
            if re.match(r"^\s*MAGMOM\s*=", line.strip()):
                f.write(mag_str + "\n")
                magmom_replaced = True
            # Replace LORBIT if it exists
            elif re.match(r"^\s*LORBIT\s*=", line.strip()):
                f.write("LORBIT = 11\n")
                lorbit_replaced = True
            # Replace ISPIN if it exists
            elif re.match(r"^\s*ISPIN\s*=", line.strip()):
                f.write("ISPIN  = 2\n")
                ispin_replaced = True
            else:
                f.write(line)

    # Append the parameters if they were not found and replaced
    with open("INCAR", "a") as f:
        if not magmom_replaced:
            f.write(mag_str + "\n")
        if not lorbit_replaced:
            f.write("LORBIT = 11\n")
        if not ispin_replaced:
            f.write("ISPIN  = 2\n")

except FileNotFoundError:
    print("INCAR file not found")
except Exception as e:
    print(f"Error: {e}")
