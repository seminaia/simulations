import numpy as np

def NR_nonlin(func, x0, es=0.0001, max_iter=50, *args):
    """
    Solves a system of nonlinear equations using the Newton-Raphson method.

    Args:
        func: A function that takes the current solution vector (x) and optional arguments (*args)
              and returns a tuple containing the Jacobian matrix (J) and the function evaluation (f).
        x0: The initial guess for the solution vector.
        es: The stopping criterion for relative error (default: 0.0001).
        max_iter: The maximum number of iterations (default: 50).
        *args: Optional arguments to be passed to the function `func`.
    
    Returns:
        x: The solution vector (or None if not converged).
        f: The function evaluation at the solution.
        ea: The final relative error (as a percentage).
        iter: The number of iterations performed.
    """

    # Set default values for optional arguments
    if len(args) < 2 or es is None:
        es = 0.0001
    if len(args) < 3 or max_iter is None:
        max_iter = 50

    x = np.array(x0)  # Convert initial guess to numpy array
    iter = 0
  
    while True:
        # Evaluate the function and Jacobian
        J, f = func(x, *args)

        # Check if Jacobian is singular (not invertible)
        if np.linalg.det(J) == 0:
            raise ValueError("Jacobian is singular at this point")

        # Solve for the update step
        dx = np.linalg.solve(J, f)

        # Update the solution vector
        x = x - dx

        # Calculate relative error
        iter += 1
        ea = 100 * np.max(np.abs(dx) / np.abs(x))

        # Check convergence criteria
        if iter >= max_iter or ea < es:
            break

    return x, f, ea, iter

if __name__ == "__main__":
    
    def test_function(x, a):
        """
        A test function for the Newton-Raphson method.

        Args:
            x: The current solution vector.
            a: A constant parameter.

        Returns:
            A tuple contaaining the Jacobian matrix (J) and the function evaluation (f).
        """
        f1 = x[0]**2 - a * x[1] + 1
        f2 = x[0] * x[1] - 2 * x[1]**2 + a
        f = np.array([f1, f2])

        # Jacobian matrix
        J = np.array([
            [2*x[0], -a],
            [x[1], x[0] - 4*x[1]]
        ])
        return J, f

    # Example usage
    x0 = np.array([1, 1])
    a = 2
    test_function(x0, a)

    x, f, ea, iter = NR_nonlin(test_function, x0, 0.001,50)
    print("Solution for test function:", x)
    print("Function evaluation at solution:", f)
    print("Relative error:", ea, "%")
    print("Iterations:", iter)
