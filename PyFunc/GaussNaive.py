import numpy as np

def GaussNaive(A, b):
  """
  Solves a system of linear equations using Gaussian Naive Elimination with back substitution.

  Args:
      A: A square numpy array representing the coefficient matrix.
      b: A numpy array representing the right-hand side vector.

  Returns:
      x: A numpy array representing the solution vector.
  """

  # Check if A is square
  m, n = A.shape
  if m != n:
    raise ValueError("Matrix A must be square")

 
  # Augmented matrix
  nb = n + 1
  Aug = np.column_stack((A, b.reshape(n, 1)))

  # Elimination loop
  for k in range(n - 1):
    for i in range(k + 1, n):
      factor = Aug[i, k] / Aug[k, k]
      Aug[i, k:nb]= Aug[i,k:nb]- factor * Aug[k, k:nb]


  # Back substitution
  x = np.zeros(n)
  x[n - 1] = Aug[n - 1, nb - 1] / Aug[n - 1, n - 1]
  for i in range(n - 2, -1, -1):
    x[i] = (Aug[i, nb - 1] - np.dot(Aug[i, i + 1:n], x[i + 1:n])) / Aug[i, i]

  return x


# Example usage
A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]])
b = np.array([4, 7, 3])
x = GaussNaive(A, b)

xt=np.linalg.solve(A,b)
print("True Solution",xt)
if x is not None:
  print("Solution:", x)