def trap(f, a, b, n=100, *args):
    """
    Trapezoidal rule integration.

    Args:
        f: The function to be integrated.
        a: The lower limit of integration.
        b: The upper limit of integration.
        n: Number of subintervals (default: 100).
        *args: Additional arguments to pass to the function f.

    Returns:
        The approximate integral of the function over the interval [a, b].
    """
    if not callable(f):
        raise ValueError("f must be a callable function")

    if not b > a:
        raise ValueError("upper bound must be greater than lower")

    x = a
    h = (b - a) / n
    s = f(a, *args)
    
    for i in range(1, n):
        x += h
        s += 2 * f(x, *args)
    
    s += f(b, *args)
    
    I = (b - a) * s / (2 * n)
    return I

if __name__ == "__main__":
    # Example usage:
    # Define the function to be integrated
    def f(x):
        return x**2
    
    # Calculate the integral of f(x) from 0 to 1 using the trapezoidal rule
    integral = trap(f, 0, 1)
    print("Approximate integral:", integral)
