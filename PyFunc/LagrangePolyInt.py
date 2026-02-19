def lagrange(x, y, xx):
  """
  Calculates the Lagrange interpolating polynomial for a given set of data points.

  Args:
      x: A list of data points (x-coordinates).
      y: A list of data points (y-coordinates).
      xx: The point at which to evaluate the polynomial.

  Returns:
      The interpolated value of the polynomial at xx.
  """

  n = len(x)
  if len(y) != n:
    raise ValueError("x and y must have the same length")

  yint = 0
  for i in range(n):
    product = y[i]
    for j in range(n):
      if i != j:
        product *= (xx - x[j]) / (x[i] - x[j])
    yint += product

  return yint

# Example usage
x = [1, 2, 3]
y = [2, 5, 7]
xx = 2.5

y_interp = lagrange(x, y, xx)
print("Interpolated value at", xx, ":", y_interp)
