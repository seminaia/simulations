def simp(f, a, b, n=100):
    """
    Simpson's 3/8 rule integration.

    Args:
        f: The function to be integrated.
        a: The lower limit of integration.
        b: The upper limit of integration.
        n: Number of subintervals (default: 100).

    Returns:
        The approximate integral of the function over the interval [a, b].
    """
    if not callable(f):
        raise ValueError("f must be a callable function")

    if not b > a:
        raise ValueError("upper bound must be greater than lower")

    h = (b - a) / n
    s = f(a)
    
    for i in range(1, n - 1, 3):
        x = a + h * i
        s += 3 * f(x)
    
    for i in range(2, n, 3):
        x = a + h * i
        s += 3 * f(x)
    
    for i in range(3, n - 1, 3):
        x = a + h * i
        s += 2 * f(x)
    
    s += f(b)
    
    I = 3 * h * s / 8
    return I

# Example usage:
# Define the function to be integrated
def f(x):
    return x**2 + 2*x+4

# Calculate the integral of f(x) from 0 to 1 using Simpson's 3/8 rule
integral = simp(f, 0, 1)
print("Approximate integral:", integral)
