import numpy as np

def simps(func, a, b, n=100):
    """
    Composite Simpson's rule integration.

    Args:
        func: The function to be integrated.
        a: The lower limit of integration.
        b: The upper limit of integration.
        n: Number of subintervals (default: 100).

    Returns:
        The approximate integral of the function over the interval [a, b].
    """
    if not callable(func):
        raise ValueError("func must be a callable function")

    if not b > a:
        raise ValueError("upper bound must be greater than lower")

    if isinstance(func, np.ndarray):
        n = len(func) - 1
        h = (b - a) / n
        xi = np.linspace(a, b, n + 1)
        integral = h / 3 * (func[0] + 2 * np.sum(func[2:-2:2]) + 4 * np.sum(func[1:-1:2]) + func[-1])
    else:
        h = (b - a) / n
        xi = np.linspace(a, b, n + 1)
        integral = h / 3 * (func(a) + 2 * np.sum(func(xi[2:-2:2])) + 4 * np.sum(func(xi[1:-1:2])) + func(b))
    
    return integral

# Example usage:
# Define the function to be integrated
def f(x):
    return x**2 + 2*x+ 4

# Calculate the integral of f(x) from 0 to 1 using Composite Simpson's rule
integral = simps(f, 0, 1)
print("Approximate integral:", integral)
