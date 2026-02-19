import numpy as np

def GaussSeidel(A, b, maxit=50, es=1e-5):
  """
  Solves a system of linear equations using Gauss-Seidel method with partial pivoting.

  Args:
      A: A square numpy array representing the coefficient matrix.
      b: A numpy array representing the right-hand side vector.
      maxit: Maximum number of iterations (default: 50).
      es: Tolerance for convergence criterion (default: 1e-5).

  Returns:
      x: A numpy array representing the solution vector.
      iterations: Number of iterations performed.
      error: Array of absolute errors for each variable at the final iteration.
  """

  # Check input arguments
  if len(A.shape) != 2 or len(b.shape) != 1 or A.shape[0] != A.shape[1]:
    raise ValueError("Matrix A must be square and dimensions of A and b must be compatible")

  m, n = A.shape

  # Augmented matrix
  Aug = np.hstack((A, b.reshape(m, 1)))

  # Partial pivoting
  for k in range(n - 1):
    ipr = np.argmax(np.abs(Aug[k:, k])) + k
    if ipr != k:
      Aug[[k, ipr], :] = Aug[[ipr, k], :]

  # Separate coefficient matrix and right-hand side vector
  A = Aug[:, :n]
  b = Aug[:, n]

  # Initialize variables
  x = np.zeros(n)
  C = np.copy(A)
  C=C.astype(float)
  for i in range(n):
    C[i, i] = 0
    x[i] = 0

  # Pre-divide by diagonal elements for efficiency
  for i in range(n):
    C[i, :] /= A[i, i]
    b[i] /= A[i, i]

  # Iteration loop
  iter = 0
  while True:
    xold = np.copy(x)
    ea = np.zeros(n)

    for i in range(n):
      x[i] = b[i] - np.dot(C[i, :], x)
      if x[i] != 0:
        ea[i] = abs((x[i] - xold[i]) / x[i]) * 100

    iter += 1
    if np.max(ea) <= es or iter >= maxit:
      break

  return x, iter, ea

# Example usage
A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]])
b = np.array([4, 7, 3])
x, iterations, error = GaussSeidel(A, b,)

print("Solution:", x)
print("Iterations:", iterations)
print("Errors:", error)
