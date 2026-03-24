import numpy as np
import matplotlib.pyplot as plt
import math

def rk4sys(dydt, tspan, y0, h, *args):
    """
    Solves a system of ordinary differential equations (ODEs) using the Runge-Kutta 4th order method.

    Args:
        dydt: Function defining the system of ODEs (takes t, y, *args as input).
        tspan: A list or array specifying the time span (start and end time).
        y0: The initial condition vector for the system.
        h: The step size for the integration.
        *args: Additional arguments to pass to the dydt function.

    Returns:
        tp: A list or array of time points.
        yp: A list or array of solution values at each time point.
    """
    # Extract start and end time
    ti = tspan[0]
    tf = tspan[-1]

    # Create time points array
    t = np.arange(ti, tf + h, h)
    num_steps = len(t)

    # Initialize solution arrays
    y = np.zeros((num_steps, len(y0)))
    y[0] = y0
    tp = t

    # Main loop for integration
    for i in range(1, num_steps):
        # Current time point
        tp0 = t[i - 1]

        # Runge-Kutta 4th order steps
        k1 = h * dydt(tp0, y[i - 1], *args)
        k2 = h * dydt(tp0 + 0.5 * h, y[i - 1] + 0.5 * k1, *args)
        k3 = h * dydt(tp0 + 0.5 * h, y[i - 1] + 0.5 * k2, *args)
        k4 = h * dydt(tp0 + h, y[i - 1] + k3, *args)

        # Update next value of y
        y[i] = y[i - 1] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return tp, y

if __name__ == "__main__":
    
    # Example usage (assuming your dydt function is defined)
    tspan =[0, 10]
    y0 = [-10,0]
    h = 0.1
    
    def sdho(t, y):
        """
        Defines the system of ordinary differential equations for a simple harmonic oscillator.
    
        Args:
            t: The current time point.
            y: A list or array containing the state variables (position and velocity).
    
        Returns:
            A list containing the derivatives of the state variables (velocity and acceleration).
        """
        m = 10.0  # Mass of the oscillator (kg)
        k = 10.0  # Spring constant (N/m)
        c = 10.0  # Viscous damping coefficient 
        
        gamma=c/(2*np.sqrt(k*m))   # damping coefficient
        omega=np.sqrt(k/m)       # Angular Frequency
    
        # Extract position and velocity from y
        position, velocity = y
    
        # Calculate acceleration (Newton's second law)
        acceleration = -2*gamma*omega * position -  (omega**2)* velocity
        
        # Return derivatives (velocity, acceleration)
        return np.array([velocity, acceleration])
    def dydt(t,y):
        return -2*y-20*t
    
    tp, yp = rk4sys(sdho, tspan, y0, h)
    
    # Use tp and yp for further analysis or plotting
    print("Time (t):\n", tp)
    print("\Position (x):\n", yp[:,0])
    print("\Velocity (v):\n", yp[:,1])
    
    plt.scatter(tp, yp[:,0], label="Position")
    plt.scatter(tp, yp[:, 1], label="Velocity")
    plt.xlabel('Time (s)')
    plt.ylabel('Position / Velocity (m, m/s)')
    plt.title('Solution (Position and Velocity) vs. Time (Runge-Kutta\'s Method)')
    plt.legend()
    plt.grid(True)
    plt.show()