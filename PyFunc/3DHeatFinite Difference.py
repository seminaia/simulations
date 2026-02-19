import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def HeatEquation(Lx, Ly, Lz, Nx, Ny, Nz, T, alpha, initial_condition, boundary_condition, dt=0.001):
    """
    Solves the 3D heat equation using the finite difference method.

    Args:
        Lx, Ly, Lz: Dimensions of the domain in the x, y, and z directions, respectively.
        Nx, Ny, Nz: Number of grid points in each dimension.
        T: Total time for simulation.
        alpha: Thermal diffusivity.
        initial_condition: Function defining the initial temperature distribution.
        boundary_condition: Function defining the boundary conditions.
        dt: Time step size (default is 0.001).

    Returns:
        U: Array containing the temperature distribution over the grid points at each time step.
        x, y, z: 1D arrays containing the grid points in the x, y, and z directions, respectively.
        t: 1D array containing the time points.
    """
    dx = Lx / (Nx - 1)  # Grid spacing in the x direction
    dy = Ly / (Ny - 1)  # Grid spacing in the y direction
    dz = Lz / (Nz - 1)  # Grid spacing in the z direction
    Nt = int(T / dt)  # Number of time steps

    # Initialize grid and solution array
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    U = np.zeros((Nx, Ny, Nz, Nt))

    # Initial condition
    U[:, :, :, 0] = initial_condition(X, Y, Z)

    # Boundary conditions
    U = boundary_condition(U)

    # Solve using finite difference method
    for t in range(Nt - 1):
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                for k in range(1, Nz - 1):
                    U[i, j, k, t + 1] = U[i, j, k, t] + alpha * dt / (dx**2) * (U[i + 1, j, k, t] + U[i - 1, j, k, t] - 2 * U[i, j, k, t]) \
                                        + alpha * dt / (dy**2) * (U[i, j + 1, k, t] + U[i, j - 1, k, t] - 2 * U[i, j, k, t]) \
                                        + alpha * dt / (dz**2) * (U[i, j, k + 1, t] + U[i, j, k - 1, t] - 2 * U[i, j, k, t])

        # Apply boundary conditions at each time step
        U = boundary_condition(U)

    # Create array of time points
    t = np.linspace(0, T, Nt)

    return U, x, y, z, t


# Define initial temperature distribution function
def initial_condition(X, Y, Z):
    return np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

# Define boundary condition function
def boundary_condition(U):
    U[0, :, :, :] = 0  # Set boundary at x=0 to zero
    U[-1, :, :, :] = 0  # Set boundary at x=Lx to zero
    U[:, 0, :, :] = 0  # Set boundary at y=0 to zero
    U[:, -1, :, :] = 0  # Set boundary at y=Ly to zero
    U[:, :, 0, :] = 0  # Set boundary at z=0 to zero
    U[:, :, -1, :] = 0  # Set boundary at z=Lz to zero
    return U


# Define parameters
Lx = Ly = Lz = 1.0  # Dimensions of the domain
Nx = Ny = Nz = 100  # Number of grid points in each dimension
T = 1  # Total time for simulation
alpha = 0.001  # Thermal diffusivity
dt = 0.001  # Time step size

# Solve the 3D heat equation
U, x, y, z, t = HeatEquation(Lx, Ly, Lz, Nx, Ny, Nz, T, alpha, initial_condition, boundary_condition, dt)

def plot_temperature(U, x, y, z, t, time_step, slice_index=None, aspect='auto'):
  """
  Plots the temperature distribution at a specific time step or along a chosen slice.

  Args:
      U: Array containing the temperature distribution.
      x, y, z: 1D arrays containing the grid points.
      t: 1D array containing the time points.
      time_step: Index of the time step to plot (if slice_index is None).
      slice_index: Index along the z-axis to plot a slice (if provided).
      aspect: Aspect ratio for the plot (default 'auto').
  """
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

  if slice_index is None:
      # Plot temperature at a specific time step
      ax.plot_surface(X[:, :, :], Y[:, :, :], U[:, :, :, time_step], rstride=1, cstride=1, cmap=plt.cm.coolwarm)
      ax.set_title('Temperature distribution at time {:.2f}'.format(t[time_step]))
  else:
      # Plot temperature slice along the z-axis
      ax.plot_surface(X[:, :, slice_index], Y[:, :, slice_index], U[:, :, slice_index, time_step], rstride=1, cstride=1, cmap=plt.cm.coolwarm)
      ax.set_title('Temperature distribution at time {:.2f} along z-axis at index {}'.format(t[time_step], slice_index))

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Temperature')
  ax.set_aspect(aspect)  # Adjust aspect ratio for better visualization
  plt.show()

# Example usage with plot function
time_step = 50  # Choose a specific time step to plot
plot_temperature(U, x, y, z, t, time_step)