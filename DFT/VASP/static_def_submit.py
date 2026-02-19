import os
import subprocess
import shutil
import sys

def modify_incar(incar_path):
    """Force ISMEAR=-5 and other essential parameters"""
    new_lines = []
    required_params = {
        "ISMEAR": "-5",
        "IBRION": "-1",
        "NSW": "0",
        "LREAL": "Auto"
    }
    
    with open(incar_path, "r") as f:
        for line in f:
            if "=" in line:
                key = line.split("=")[0].strip().upper()
                if key in required_params:
                    new_lines.append(f"{key} = {required_params[key]}\n")
                    del required_params[key]
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
    
    # Add any missing required parameters
    for key, value in required_params.items():
        new_lines.append(f"{key} = {value}\n")
    
    with open(incar_path, "w") as f:
        f.writelines(new_lines)

def adjust_kpoints(vasp_dir):
    """Ensure automatic k-point grids have >4 points"""
    kpoints_path = os.path.join(vasp_dir, "KPOINTS")
    if not os.path.exists(kpoints_path):
        raise FileNotFoundError(f"KPOINTS not found in {vasp_dir}")

    with open(kpoints_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 4 or lines[1].strip() != "0":
        return  # Only handle automatic grids

    # Get original grid
    grid = list(map(int, lines[3].split()[:3]))
    original_grid = grid.copy()
    
    # Adjust grid to ensure minimum 2x2x2
    new_grid = [max(d, 2) for d in grid]
    
    if new_grid != grid:
        lines[3] = " ".join(map(str, new_grid)) + "\n"
        with open(kpoints_path, "w") as f:
            f.writelines(lines)
        print(f"Adjusted KPOINTS grid {original_grid} → {new_grid}")

def check_kpoints(vasp_dir):
    """Verify k-point configuration"""
    kpoints_path = os.path.join(vasp_dir, "KPOINTS")
    with open(kpoints_path, "r") as f:
        lines = f.readlines()
    
    if len(lines) < 4:
        return
    
    mode = lines[1].strip()
    if mode == "0":
        grid = list(map(int, lines[3].split()[:3]))
        nkpt = grid[0] * grid[1] * grid[2]
        print(f"Automatic k-point grid: {grid} ({nkpt} points)")
    elif mode == "1":
        try:
            nkpt = int(lines[2].strip())
            if nkpt <= 4:
                print(f"WARNING: Only {nkpt} explicit k-points - may need manual adjustment!")
        except:
            pass

def submit_job(directory):
    """Handle job submission with error checking"""
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        job_name = os.path.basename(os.path.dirname(directory))
        result = subprocess.run(
            ["sbatch", "--job-name", job_name, "job.sh"],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Submitted {job_name}: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Submission failed: {e.stderr.strip()}")
        if "IBZKPT" in e.stderr:
            print("  TIP: Check k-point configuration in KPOINTS file")
    finally:
        os.chdir(cwd)

def main():
    # Validate required files
    required_files = ["job.sh", "update_mag.py"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"Missing required files: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Process directories
    for entry in os.listdir():
        vasp_dir = os.path.join(entry, "vasp_gam")
        if not os.path.isdir(vasp_dir):
            continue

        print(f"\n{'#'*40}")
        print(f"Processing {entry}")
        
        try:
            # 1. INCAR modifications
            incar_path = os.path.join(vasp_dir, "INCAR")
            if not os.path.exists(incar_path):
                print("Skipping - Missing INCAR")
                continue
            modify_incar(incar_path)
            print("Updated INCAR with ISMEAR=-5")

            # 2. KPOINTS adjustments
            adjust_kpoints(vasp_dir)
            check_kpoints(vasp_dir)

            # 3. File management
            for f in required_files:
                shutil.copy(f, vasp_dir)
            print("Copied required files")

            # 4. Structure updates
            contcar = os.path.join(vasp_dir, "CONTCAR")
            if os.path.exists(contcar):
                shutil.copy(contcar, os.path.join(vasp_dir, "POSCAR"))
                print("Updated POSCAR from CONTCAR")

            # 5. Magnetic moments
            subprocess.run(
                ["python3", "update_mag.py"],
                cwd=vasp_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print("Ran magnetic moment update")

            # 6. Job submission
            submit_job(vasp_dir)

        except Exception as e:
            print(f"Error processing {entry}: {str(e)}")
            continue

if __name__ == "__main__":
    main()