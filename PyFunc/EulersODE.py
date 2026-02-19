import numpy as np
import matplotlib.pyplot as plt

def eulode(dydt, tspan, y0, h, *args):
    """
    Solves an ODE using Euler's method.

    Args:
        dydt: The function representing the derivative of y.
        tspan: A tuple containing the initial and final time points (ti, tf).
        y0: The initial value of y.
        h: The step size.
        *args: Additional arguments to pass to dydt.

    Returns:
        t: An array of time values.
        y: An array of the corresponding y values.
    """

    if len(args) == 0:
        args = ()  # Handle potential issue with *args being empty

    if len(tspan) != 2:
        raise ValueError("tspan must be a tuple of length 2")

    ti, tf = tspan
    if tf <= ti:
        raise ValueError("tf must be greater than ti")

    t = np.arange(ti, tf + h, h)  # Create time array with proper inclusion of tf
    n = len(t)

    y = y0 * np.ones(n)  # Initialize y array with y0
    for i in range(n - 1):
        y[i + 1] = y[i] + dydt(t[i], y[i], *args) * h

    return t, y

def dydt(y,t):
    return -2*y**2-2*y*t

tspan=(0,5)
y0=1

t, y=eulode(dydt,tspan,y0,0.1)
print("Time (t):\n",t)
print("\nSolution (y):\n",y)

plt.plot(t,y,label="y")

plt.xlabel('Time (t)')
plt.ylabel('y')
plt.title('Solution (y) vs. Time (Euler\'s Method)')
plt.legend()
plt.grid(True)
plt.show()