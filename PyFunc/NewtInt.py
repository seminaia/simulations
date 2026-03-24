def newton_polynomial(x, y, xx):
  """
  Calculates the Newton interpolating polynomial for a given set of data points.

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

  # Initialize divided difference table
  b = [[0] * n for _ in range(n)]
  for i in range(n):
    b[i][0] = y[i]

  # Build the divided difference table
  for j in range(1, n):
    for i in range(n - j):
      b[i][j] = (b[i - 1][j - 1] - b[i][j]) / (x[i + j] - x[i])

  # Evaluate the polynomial
  xt = 1
  yint = b[0][0]
  for j in range(1, n):
    xt *= (xx - x[j])
    yint += b[0][j] * xt

  return yint
if __name__ == "__main__":
  
  # Example usage
  x = [1, 2, 3]
  y = [2, 5, 7]
  xx = 2.5
  
  y_interp = newton_polynomial(x, y, xx)
  print("Interpolated value at", xx, ":", y_interp)
