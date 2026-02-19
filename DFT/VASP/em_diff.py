import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix
import pandas as pd
import json
from tqdm import tqdm
import unittest
import logging
from concurrent.futures import ThreadPoolExecutor
from scipy.special import erf

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DiffusionDriftSolver:
    def __init__(self, D, v, L, J0, C0, Nx, Nt, 
                 boundary_conditions=None,
                 gaussian_peak=0.5, 
                 gaussian_center=0.05, 
                 gaussian_width=0.01):
        # Existing parameters
        self.D = D  
        self.v = v  
        self.L = L  
        self.J0 = J0  
        self.C0 = C0  
        self.Nx = Nx  
        self.Nt = Nt  
        self.boundary_conditions = boundary_conditions or {"type": "dirichlet", "values": (C0, C0)}
        self.dx = L / (Nx - 1)
        self.dt = self.dx**2 / (4 * D)

        # ERF initial condition parameters
        #self.C_max = C_max if C_max is not None else C0  # Default to C0
        #self.x0 = x0 if x0 is not None else 0.0  # Default to x=0
        #self.sigma = sigma if sigma is not None else L/10  # Default to L/10
        
    # Gaussian initial condition parameters
        self.gaussian_peak = gaussian_peak
        self.gaussian_center = gaussian_center
        self.gaussian_width = gaussian_width
        

    #def erf_initial_condition(self, x: np.ndarray) -> np.ndarray:
    #    """Error function initial condition using instance parameters."""
    #    return self.C_max * (1 + erf((x - self.x0) / (self.sigma * np.sqrt(2))))
    
    def gaussian_initial_condition(self, x):
        return self.gaussian_peak * np.exp(-(x - self.gaussian_center)**2 / 
                                     (2 * self.gaussian_width**2)) + self.C0
        
    def compute_mode_coefficient(self, x: np.ndarray, n: int) -> float:
        """Compute Fourier coefficient using stored initial condition."""
        integrand = (self.gaussian_initial_condition(x) - self.steady_state_solution(x)) * np.sin(n*np.pi*x/self.L)
        return 2/self.L * np.trapz(integrand, x)
    
    def compute_flux_at_x0(self, c_steady):
        """Compute flux at x=0 using 4th-order finite difference."""
        dc_dx_at_x0 = (-c_steady[2] + 8*c_steady[1] - 8*c_steady[0] + c_steady[-1]) / (12 * self.dx)
        return -self.D * dc_dx_at_x0 + self.v * c_steady[0]

    def validate_parameters(self):
        if self.dt > self.dx**2 / (4 * self.D):
            raise ValueError("Time step exceeds stability limit.")

    def steady_state_solution(self, x: np.ndarray) -> np.ndarray:
        """Steady-state solution with flux boundary conditions."""
        # General solution: C(x) = A + B*exp(vx/D)
        # Apply flux BC at x=0: -D*C'(0) + v*C(0) = J0
        # Apply C(L) = C0 (Dirichlet at x=L)
        A = self.J0 / self.v
        B = (self.C0 - A) / (np.exp(self.v*self.L/self.D) - 1)
        return A + B * np.exp(self.v * x / self.D)

    def transient_solution(self, x: np.ndarray, t: float) -> np.ndarray:
        """Transient solution using eigenfunction expansion."""
        # Eigenfunctions satisfy homogeneous BCs
        lambda_n = np.array([(n*np.pi/self.L)**2 * self.D for n in range(1, 10)])
        coeffs = np.array([self.compute_mode_coefficient(x, n) for n in range(1, 10)])
        return np.sum([c * np.exp(-lam*t) * np.sin((n*np.pi/self.L)*x) 
                        for n, (c, lam) in enumerate(zip(coeffs, lambda_n), start=1)], axis=0)

    def combined_solution(self, x: np.ndarray, t: float) -> np.ndarray:
        return self.steady_state_solution(x) + self.transient_solution(x, t)

    def apply_boundary_conditions(self, c):
        """Handle all BC types: dirichlet, neumann, robin"""
        bc_type = self.boundary_conditions["type"].lower()
        
        # Left boundary (x=0)
        if bc_type == "dirichlet":
            c[0] = self.boundary_conditions["values"][0]
        elif bc_type == "neumann":
            c[0] = c[1]  # Simple first-order approximation
        elif bc_type == "robin":
            # -D*dc/dx + v*c = J0 using ghost cell (2nd order accurate)
            c[0] = (4*c[1] - c[2] + 2*self.dx*self.J0/self.D) / 3
        
        # Right boundary (x=L) - always Dirichlet
        c[-1] = self.C0
        
    def plot_results(self, x, analytical_results, time_points, concentration_at_x0, fourier_numbers, discriminant_cases=False):
        """
        Plot and save results.
        """
        plt.figure(figsize=(18, 6))
        concentration_matrix = np.array(analytical_results)  # Convert to numpy array

        # Spatial evolution plot
        plt.subplot(1, 3, 1)
        step = len(analytical_results) // 11 or 1
        for i in range(0, len(analytical_results), step):
            plt.plot(x, analytical_results[i], label=f"t = {i * self.dt:.2e} s")
        plt.xlabel("Position (cm)")
        plt.ylabel("Concentration (mol/cm³)")
        plt.title("Concentration Over Space")
        plt.legend()
        plt.grid()

        # Temporal evolution plot
        plt.subplot(1, 3, 2)
        plt.plot(time_points, concentration_at_x0, label="Concentration at x = 0", color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("Concentration (mol/cm³)")
        plt.title("Concentration Over Time at x = 0")
        plt.legend()
        plt.grid()
    
        # Fourier number plot
        plt.subplot(1, 3, 3)
        plt.plot(time_points, fourier_numbers, label="Fourier Number", color="green")
        plt.xlabel("Time (s)")
        plt.ylabel("Fourier Number")
        plt.title("Fourier Number Over Time")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig("main_plot.png", dpi=300, bbox_inches="tight")
        plt.show()

        if discriminant_cases:
            plt.figure(figsize=(12, 6))
            t = 1
            # Create temporary solver instances for different cases
            cases = [
                ("Distinct Real Roots", 1e-10, 1e-5),
                ("Repeated Real Roots", 1e-10, 2 * np.sqrt(self.D * (self.D * (np.pi/self.L)**2 + 1e-6 * (np.pi/self.L)))),
                ("Complex Roots", 1e-10, 1e-7)
            ]

            for label, D_case, v_case in cases:
                case_solver = DiffusionDriftSolver(
                    D=D_case, v=v_case, L=self.L,
                    J0=self.J0, C0=self.C0,
                    Nx=self.Nx, Nt=self.Nt,
                    boundary_conditions=self.boundary_conditions
                )
                c_transient = case_solver.transient_solution(x, t)
                plt.plot(x, c_transient, label=label)

            plt.xlabel("Position (cm)")
            plt.ylabel("Concentration (mol/cm³)")
            plt.title("Transient Solution for Different Discriminant Cases")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig("discriminant_cases_plot.png", dpi=300, bbox_inches="tight")
            plt.show()

            # 3D Surface plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            X, T = np.meshgrid(x, time_points)
            concentration_matrix = np.array(analytical_results)
            surf = ax.plot_surface(X, T, concentration_matrix, cmap='viridis', edgecolor='none')
            ax.set_xlabel("Position (cm)")
            ax.set_ylabel("Time (s)")
            ax.set_zlabel("Concentration (mol/cm³)")
            ax.set_title("Combined Solution: Concentration Over Space and Time")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Concentration (mol/cm³)")
            plt.savefig("combined_solutions.png", dpi=300, bbox_inches="tight")
            plt.tight_layout()
            plt.show()


# Unit tests
class TestDiffusionDriftSolver(unittest.TestCase):
    def setUp(self):
        self.solver = DiffusionDriftSolver(D=1e-6, v=1e-4, L=1e-2, J0=1, C0=1, Nx=200, Nt=1000)

    def test_steady_state(self):
        x = np.linspace(0, self.solver.L, self.solver.Nx)
        c_steady = self.solver.steady_state_solution(x)
        flux_at_0 = self.solver.compute_flux_at_x0(c_steady)
        self.assertTrue(
            np.isclose(flux_at_0, self.solver.J0, atol=1e-3),
            f"Flux boundary condition failed. Expected {self.solver.J0}, got {flux_at_0}"
        )

    def test_combined_solution(self):
        x = np.linspace(0, self.solver.L, self.solver.Nx)
        c_combined = self.solver.combined_solution(x, 1e10)
        c_steady = self.solver.steady_state_solution(x)
        self.assertTrue(np.allclose(c_combined, c_steady, atol=1e-4),
                        "Combined solution does not match steady state.")

if __name__ == "__main__":
    # Initialize solver with Dirichlet boundary conditions
    boundary_conditions = {"type": "robin", "values": (1e-4, 1e-4)}
    solver = DiffusionDriftSolver(D=1e-6, v=1e-4, L=1e-2, J0=1, C0=1, Nx=200, Nt=5000, boundary_conditions=boundary_conditions)

    # Parameters for error function initial condition
    C_max = solver.C0  # Maximum concentration
    x0 = solver.L / 2  # Initial position of the error function peak
    sigma = solver.L / 10  # Initial width of the error function
    # Generate initial condition
    x = np.linspace(0, solver.L, solver.Nx)
    C_initial = solver.gaussian_initial_condition(x)  # No arguments needed!
    # Compute analytical solution over time
    time_points = np.arange(0, solver.Nt * solver.dt, solver.dt)
    fourier_numbers = solver.D * time_points / solver.L**2

    # Parallel computation of analytical results
    with ThreadPoolExecutor() as executor:
        analytical_results = list(executor.map(
            lambda t: np.maximum(solver.combined_solution(x, t), 0),
            tqdm(time_points, desc="Computing solutions")
        ))

    # Apply boundary conditions to each result
    for result in analytical_results:
        solver.apply_boundary_conditions(result)

    # Extract concentration at x = 0
    concentration_at_x0 = [result[0] for result in analytical_results]

    # Save results
    metadata = {
        "D": {"value": solver.D, "unit": "cm^2/s"},
        "v": {"value": solver.v, "unit": "cm/s"},
        "L": {"value": solver.L, "unit": "cm"},
        "Pe": solver.v * solver.L / solver.D,  # Péclet number
        "J0": {"value": solver.J0, "unit": "mol/cm^2/s"},
        "C0": {"value": C_initial.tolist(), "unit": "mol/cm^3"},
        "Nx": solver.Nx,
        "Nt": solver.Nt,
        "dx": {"value": solver.dx, "unit": "cm"},
        "dt": {"value": solver.dt, "unit": "s"},
        "boundary_conditions": solver.boundary_conditions
    }

    # Save analytical results to CSV
    pd.DataFrame(analytical_results).to_csv("analytical_results.csv", index=False)

    # Save concentration at x = 0 over time to CSV
    pd.DataFrame({"Time": time_points, "Concentration": concentration_at_x0}).to_csv("concentration_at_x0.csv", index=False)

    # Save metadata to JSON
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Log Péclet number information
    Pe = metadata["Pe"]
    if Pe > 1:
        logging.info(f"High Péclet number detected: Pe = {Pe:.2f}. Advection-dominated transport.")
    else:
        logging.info(f"Low Péclet number detected: Pe = {Pe:.2f}. Diffusion-dominated transport.")

    # Plot results
    solver.plot_results(x, analytical_results, time_points, concentration_at_x0,fourier_numbers, discriminant_cases=True)