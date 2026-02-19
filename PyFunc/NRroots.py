def newton_raphson(func, dfunc, x0, es=0.0001, max_iter=50, *args):
  """
  Solves for the root of a function using the Newton-Raphson method.

  Args:
      func: A function that takes a single argument (x) and returns the function evaluation (f(x)).
      dfunc: A function that takes a single argument (x) and returns the derivative of the function (f'(x)).
      x0: The initial guess for the root.
      es: The stopping criterion for relative error (default: 0.0001).
      max_iter: The maximum number of iterations (default: 50).
      *args: Optional arguments to be passed to the function `func` and `dfunc`.

  Returns:
      root: The approximate root of the function.
      ea: The final relative error (as a percentage).
      iter: The number of iterations performed.
  """

    #  if len(args) < 2:
    #raise ValueError("At least 3 input arguments are required (function, derivative, and initial guess)")

  iter = 0
  while True:
    x_old = x0
    x0 = x0 - func(x0) / dfunc(x0)
    iter += 1

    if x0 != 0:
      ea = abs((x0 - x_old) / x0) * 100
    else:
      ea = float('inf')  # Handle division by zero

    if ea <= es or iter >= max_iter:
      break

  return x0, ea, iter

# Example usage
def f(x):
  return x**2 - 2  # Replace with your actual function

print(f)

def df(x):
  return 2*x  # Replace with your actual derivative

x0 = 1.5  # Initial guess
es = 0.01  # Stopping criteria (optional)
max_iter = 100  # Maximum iterations (optional)

root, ea, iter = newton_raphson(f, df, x0, es, max_iter)
print("Root:", root)
print("Relative error:", ea, "%")
print("Iterations:", iter)
