from lammps import lammps
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time


class AluminumLAMMPS_Simulation:
    """
    Aluminum FCC MD using LAMMPS Python interface.

    Key fixes vs your original:
      - Set a realistic timestep for units metal (default is NOT safe): timestep 0.001 ps (1 fs)
      - Do NOT run fix nve and fix nvt at the same time (double integration)
      - Use a physically meaningful density/volume evolution via NPT (optional but recommended)
      - Use global atom count (get_natoms) instead of nlocal (MPI-safe)
      - Cleaner thermo logging and parsing
    """

    def __init__(self, lattice_constant=4.05, nx=5, ny=5, nz=5, seed=4928459):
        self.lattice_constant = float(lattice_constant)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.seed = int(seed)

        self.L = None
        self.natoms = None

        # Physical constants
        self.avogadro = 6.02214076e23  # mol^-1
        self.al_mass = 26.9815385      # g/mol

    # -------------------------
    # LAMMPS setup
    # -------------------------
    def setup_simulation(self):
        print("=" * 70)
        print("Aluminum LAMMPS Simulation")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        # Quiet screen, write a log
        cmdargs = ["-screen", "none", "-log", "lammps.log"]
        self.L = lammps(cmdargs=cmdargs)

        self.L.command("clear")
        self.L.command("units metal")
        self.L.command("atom_style atomic")
        self.L.command("atom_modify map array sort 0 0")

        # IMPORTANT: metal units timestep is in ps.
        # 0.001 ps = 1 fs (typical for EAM Al).
        self.L.command("timestep 0.001")

        # Boundary (periodic bulk)
        self.L.command("boundary p p p")

    def create_structure(self):
        a = self.lattice_constant
        nx, ny, nz = self.nx, self.ny, self.nz

        # Use lattice/region in lattice units for clean FCC fill
        print("\nCreating aluminum FCC block:")
        print(f"  Lattice constant: {a:.4f} Å")
        print(f"  Unit cells: {nx} x {ny} x {nz}")

        self.L.command(f"lattice fcc {a}")
        self.L.command(f"region box block 0 {nx} 0 {ny} 0 {nz} units lattice")
        self.L.command("create_box 1 box")
        self.L.command("create_atoms 1 box")
        self.L.command(f"mass 1 {self.al_mass}")

        # Global atom count (MPI-safe)
        self.natoms = int(self.L.get_natoms())
        print(f"  Total atoms created: {self.natoms}")

        # Theoretical density from initial box volume
        self.L.command("variable v0 equal vol")
        vol0 = float(self.L.extract_variable("v0", None, 0))  # Å^3
        rho0 = self.calculate_density(vol0)
        print(f"  Initial box volume: {vol0:.2f} Å³")
        print(f"  Initial density: {rho0:.3f} g/cm³ (at initial a)")

    def setup_potential(self):
        print("\nSetting up interatomic potential...")

        # Your original choice (OpenKIM model). This requires KIM support + the model installed.
        # If this line errors, swap to a local EAM file, e.g.:
        #   pair_style eam/alloy
        #   pair_coeff * * Al99.eam.alloy Al
        self.L.command("pair_style kim EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005")
        self.L.command("pair_coeff * * Al")

        self.L.command("neighbor 0.3 bin")
        self.L.command("neigh_modify every 20 delay 0 check yes")

    def setup_computes(self):
        self.L.command("compute cTemp all temp")
        self.L.command("compute cPE   all pe")
        self.L.command("compute cKE   all ke")

        # Convenience variables
        self.L.command("variable vPress equal press")
        self.L.command("variable vVol   equal vol")
        self.L.command("variable vEtot  equal etotal")
        self.L.command("variable vTemp  equal temp")
        self.L.command("variable vPe    equal pe")
        self.L.command("variable vKe    equal ke")

    def setup_thermo(self, thermo_freq=100, thermo_file="thermo.out"):
        # Standard thermo output
        self.L.command(f"thermo {int(thermo_freq)}")
        self.L.command("thermo_style custom step temp pe ke etotal press vol density")
        self.L.command("thermo_modify lost ignore flush yes")

        # File logging via fix print (easy parsing)
        # Columns: step temp pe ke etotal press vol density
        self.L.command(
            f"fix fThermo all print {int(thermo_freq)} "
            "'$(step) $(temp) $(pe) $(ke) $(etotal) $(press) $(vol) $(density)' "
            f"file {thermo_file} screen no"
        )

    # -------------------------
    # Runs
    # -------------------------
    def energy_minimization(self):
        print("\n" + "=" * 70)
        print("Energy Minimization")
        print("=" * 70)

        self.L.command("reset_timestep 0")
        self.L.command("min_style cg")
        self.L.command("minimize 1.0e-10 1.0e-12 2000 20000")

        pe = float(self.L.extract_compute("cPE", 0, 0))
        print(f"Potential energy after minimization: {pe:.6f} eV")
        print(f"PE per atom: {pe / self.natoms:.6f} eV/atom")
    
    def thermal_equilibration_nvt(self, temp=300, steps=2000, tdamp=0.1):
        """
        NVT equilibration at fixed volume.
        """
        print("\n" + "=" * 70)
        print(f"NVT Equilibration @ {temp} K")
        print("=" * 70)

        self.L.command("reset_timestep 0")
        self.L.command(f"velocity all create {float(temp)} {self.seed} rot yes dist gaussian")

        self.L.command(
            f"fix NVT all nvt temp {float(temp)} {float(temp)} {float(tdamp)}"
        )
        self.L.command(f"run {int(steps)}")
        self.L.command("unfix NVT")


    def pressure_equilibration_npt(self, temp=300, press=0.0, steps=5000, tdamp=0.1, pdamp=1.0):
        """
        NPT equilibration so the box (and density) can relax/expand properly.
        For bulk Al, iso is typical (unless you need anisotropic).
        """
        print("\n" + "=" * 70)
        print(f"NPT Equilibration @ {temp} K, P={press} bar (iso)")
        print("=" * 70)

        # Start from current velocities (after NVT), or create if none
        # (If you want always new velocities, uncomment velocity create.)
        # self.L.command(f"velocity all create {float(temp)} {self.seed} rot yes dist gaussian")

        self.L.command("reset_timestep 0")
        self.L.command("fix fNPT all npt temp {} {} {} iso {} {} {}".format(
            float(temp), float(temp), float(tdamp),
            float(press), float(press), float(pdamp)
        ))
        self.L.command(f"run {int(steps)}")
        self.L.command("unfix fNPT")

    def production_run(self, ensemble="nve", temp=300, press=0.0, steps=10000, tdamp=0.1, pdamp=1.0):
        """
        Production run:
          - ensemble="nve": energy-conserving dynamics (volume fixed to current)
          - ensemble="nvt": constant T dynamics
          - ensemble="npt": constant T,P dynamics
        """
        ensemble = ensemble.lower().strip()
        print("\n" + "=" * 70)
        print(f"Production Run ({ensemble.upper()})")
        print("=" * 70)

        self.L.command("reset_timestep 0")

        if ensemble == "nve":
            self.L.command("fix fPROD all nve")
        elif ensemble == "nvt":
            self.L.command("fix fPROD all nvt temp {} {} {}".format(float(temp), float(temp), float(tdamp)))
        elif ensemble == "npt":
            self.L.command("fix fPROD all npt temp {} {} {} iso {} {} {}".format(
                float(temp), float(temp), float(tdamp),
                float(press), float(press), float(pdamp)
            ))
        else:
            raise ValueError("ensemble must be one of: 'nve', 'nvt', 'npt'")

        self.L.command(f"run {int(steps)}")
        self.L.command("unfix fPROD")

    # -------------------------
    # Analysis / IO
    # -------------------------
    def calculate_density(self, volume_angstrom3):
        """Density in g/cm^3 from volume in Å^3 and fixed atom count."""
        volume_cm3 = float(volume_angstrom3) * 1e-24
        mass_per_atom_g = self.al_mass / self.avogadro
        total_mass_g = self.natoms * mass_per_atom_g
        return total_mass_g / volume_cm3

    def collect_results(self):
        print("\n" + "=" * 70)
        print("Final Results")
        print("=" * 70)

        temp = float(self.L.extract_compute("cTemp", 0, 0))
        pe = float(self.L.extract_compute("cPE", 0, 0))
        ke = float(self.L.extract_compute("cKE", 0, 0))
        press = float(self.L.extract_variable("vPress", None, 0))
        vol = float(self.L.extract_variable("vVol", None, 0))

        density_manual = self.calculate_density(vol)

        print(f"Final temperature: {temp:.2f} K")
        print(f"Final PE: {pe:.6f} eV  ({pe / self.natoms:.6f} eV/atom)")
        print(f"Final KE: {ke:.6f} eV  ({ke / self.natoms:.6f} eV/atom)")
        print(f"Final total E: {pe + ke:.6f} eV  ({(pe + ke) / self.natoms:.6f} eV/atom)")
        print(f"Pressure: {press:.2f} bar")
        print(f"Volume: {vol:.2f} Å³")
        print(f"Density (manual): {density_manual:.3f} g/cm³")

        # Sample atom positions
        natoms = int(self.L.get_natoms())
        if natoms > 0:
            x = self.L.numpy.extract_atom("x")
            print("\nSample atom positions (first 5):")
            for i in range(min(5, natoms)):
                print(f"  Atom {i + 1}: ({x[i][0]:.3f}, {x[i][1]:.3f}, {x[i][2]:.3f}) Å")

        return {
            "temperature": temp,
            "pe": pe,
            "ke": ke,
            "pressure": press,
            "volume": vol,
            "natoms": natoms,
            "density": density_manual,
            "energy_per_atom": (pe + ke) / natoms,
            "pe_per_atom": pe / natoms
        }
    
    def analyze_structure(self):
        print("\n" + "=" * 70)
        print("Structural Analysis")
        print("=" * 70)

        # Correct unpacking of extract_box()
        box = self.L.extract_box()

        # box[0] = [xlo, ylo, zlo]
        # box[1] = [xhi, yhi, zhi]
        lo = box[0]
        hi = box[1]

        xlo, ylo, zlo = lo
        xhi, yhi, zhi = hi

        lx = xhi - xlo
        ly = yhi - ylo
        lz = zhi - zlo

        print(f"Box bounds:")
        print(f"  x = [{xlo:.4f}, {xhi:.4f}] Å")
        print(f"  y = [{ylo:.4f}, {yhi:.4f}] Å")
        print(f"  z = [{zlo:.4f}, {zhi:.4f}] Å")

        print(f"Box lengths:")
        print(f"  Lx = {lx:.4f} Å")
        print(f"  Ly = {ly:.4f} Å")
        print(f"  Lz = {lz:.4f} Å")

        # Effective lattice parameter (assuming cubic-ish)
        a_eff = lx / self.nx
        print(f"\nEffective lattice parameter from Lx/nx: {a_eff:.6f} Å")
        print(f"Δa/a0 = {((a_eff / self.lattice_constant) - 1.0) * 100:.4f}%")


    def save_restart(self, filename="al_restart.restart"):
        self.L.command(f"write_restart {filename}")
        print(f"\nRestart file saved as '{filename}'")

    def save_data_file(self, filename="al_final.data"):
        self.L.command(f"write_data {filename}")
        print(f"Data file saved as '{filename}'")

    def read_thermo_file(self, thermo_file="thermo.out"):
        """
        Reads thermo.out written by fix print with columns:
          step temp pe ke etotal press vol density
        Returns arrays or (None,...).
        """
        if not os.path.exists(thermo_file) or os.path.getsize(thermo_file) == 0:
            print(f"\nThermo file '{thermo_file}' not found or empty.")
            return (None,) * 8

        data = []
        with open(thermo_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                try:
                    data.append([float(x) for x in parts[:8]])
                except ValueError:
                    continue

        if not data:
            print(f"\nNo numeric data found in '{thermo_file}'.")
            return (None,) * 8

        arr = np.array(data, dtype=float)
        steps = arr[:, 0]
        temp = arr[:, 1]
        pe = arr[:, 2]
        ke = arr[:, 3]
        etot = arr[:, 4]
        press = arr[:, 5]
        vol = arr[:, 6]
        dens = arr[:, 7]
        return steps, temp, pe, ke, etot, press, vol, dens

    def plot_results(self, thermo_file="thermo.out", tail=200):
        steps, temp, pe, ke, etot, press, vol, dens = self.read_thermo_file(thermo_file)
        if steps is None:
            print("\nNo thermo data available for plotting.")
            return

        print("\nCreating plots...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].plot(steps, temp, linewidth=1.2, alpha=0.8)
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Temperature (K)")
        axes[0, 0].set_title("Temperature")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, pe, linewidth=1.2, alpha=0.8)
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Potential Energy (eV)")
        axes[0, 1].set_title("Potential Energy")
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(steps, ke, linewidth=1.2, alpha=0.8)
        axes[0, 2].set_xlabel("Step")
        axes[0, 2].set_ylabel("Kinetic Energy (eV)")
        axes[0, 2].set_title("Kinetic Energy")
        axes[0, 2].grid(True, alpha=0.3)

        axes[1, 0].plot(steps, etot, linewidth=1.2, alpha=0.8)
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Total Energy (eV)")
        axes[1, 0].set_title("Total Energy")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(steps, press, linewidth=1.2, alpha=0.8)
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Pressure (bar)")
        axes[1, 1].set_title("Pressure")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(steps, dens, linewidth=1.2, alpha=0.8)
        axes[1, 2].axhline(y=2.70, linestyle=":", linewidth=1.2, label="~2.70 g/cm³ (RT)")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_ylabel("Density (g/cm³)")
        axes[1, 2].set_title("Density")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()

        plt.tight_layout()
        plt.savefig("al_simulation_results.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Stats over tail window
        n = min(int(tail), len(steps))
        print("\n" + "=" * 70)
        print(f"Statistics over last {n} records:")
        print(f"  T:   {np.mean(temp[-n:]):.2f} ± {np.std(temp[-n:]):.2f} K")
        print(f"  PE:  {np.mean(pe[-n:]):.3f} ± {np.std(pe[-n:]):.3f} eV")
        print(f"  KE:  {np.mean(ke[-n:]):.3f} ± {np.std(ke[-n:]):.3f} eV")
        print(f"  Etot:{np.mean(etot[-n:]):.3f} ± {np.std(etot[-n:]):.3f} eV")
        print(f"  P:   {np.mean(press[-n:]):.2f} ± {np.std(press[-n:]):.2f} bar")
        print(f"  ρ:   {np.mean(dens[-n:]):.3f} ± {np.std(dens[-n:]):.3f} g/cm³")
        print("=" * 70)

    def cleanup_files(self, files=("thermo.out", "lammps.log")):
        for f in files:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed {f}")

    # -------------------------
    # Main workflow
    # -------------------------
    def run_complete_simulation(
        self,
        temp=300.0,
        # equilibration
        nvt_steps=2000,
        npt_steps=5000,
        press=0.0,
        # production
        prod_ensemble="nve",
        prod_steps=10000,
        thermo_freq=100,
        cleanup=False
    ):
        start_time = time.time()

        try:
            self.setup_simulation()
            self.create_structure()
            self.setup_potential()
            self.setup_computes()
            self.setup_thermo(thermo_freq=thermo_freq, thermo_file="thermo.out")

            self.energy_minimization()

            # Thermalize at fixed volume first
            self.thermal_equilibration_nvt(temp=temp, steps=nvt_steps, tdamp=0.1)

            # Relax volume/density at target T,P (recommended)
            self.pressure_equilibration_npt(temp=temp, press=press, steps=npt_steps, tdamp=0.1, pdamp=1.0)

            # Production run (choose nve/nvt/npt)
            self.production_run(
                ensemble=prod_ensemble,
                temp=temp,
                press=press,
                steps=prod_steps,
                tdamp=0.1,
                pdamp=1.0
            )

            results = self.collect_results()
            self.analyze_structure()
            self.save_restart()
            self.save_data_file()
            self.plot_results("thermo.out")

            if cleanup:
                self.cleanup_files()

            elapsed = time.time() - start_time
            print("\n" + "=" * 70)
            print("Simulation completed successfully!")
            print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Elapsed time: {elapsed:.2f} seconds")
            print("=" * 70)

            return results

        except Exception as e:
            print(f"\nError during simulation: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            if self.L is not None:
                self.L.close()


if __name__ == "__main__":
    print("Starting Aluminum LAMMPS Simulation...")
    print("=" * 70)

    sim = AluminumLAMMPS_Simulation(
        lattice_constant=4.05,
        nx=5, ny=5, nz=5
    )

    # Suggested defaults:
    # - NVT 2000 steps (2 ps)
    # - NPT 5000 steps (5 ps) to relax density at 300 K, 0 bar
    # - Production NVE or NVT; if you want stable density/pressure statistics, use NPT production too.
    results = sim.run_complete_simulation(
        temp=300.0,
        nvt_steps=2000,
        npt_steps=5000,
        press=1.01325,
        prod_ensemble="nve",   # or "nvt" / "npt"
        prod_steps=10000,
        thermo_freq=100,
        cleanup=False
    )

    if results:
        print("\nFinal Summary:")
        print(f"  Atoms: {results['natoms']}")
        print(f"  Energy per atom (total): {results['energy_per_atom']:.6f} eV/atom")
        print(f"  PE per atom: {results['pe_per_atom']:.6f} eV/atom")
        print(f"  Density (manual): {results['density']:.3f} g/cm³")
