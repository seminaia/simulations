




def bisect(func, xl, xu, es=0.0001, maxit=100, *args):
   """
   Performs the bisection method to find a root of a function.

   Args:
       func: The function for which to find a root.
       xl: The lower bound of the search interval.
       xu: The upper bound of the search interval.
       es: The desired tolerance for the error (default = 0.0001).
       maxit: The maximum number of iterations (default = 100).
       *args: Additional arguments to pass to the function.

   Returns:
       root: The approximate root of the function.
       fx: The value of the function at the root.
       ea: The approximate relative error.
       iter: The number of iterations performed.
   """

   if len(args) == 0:  # Handle potential issue with *args being empty
       args = ()

   if not callable(func):
       raise TypeError("func must be a callable function")

   if not isinstance(xl, (int, float)) or not isinstance(xu, (int, float)):
       raise TypeError("xl and xu must be numerical values")

   if not isinstance(es, (int, float)):
       raise TypeError("es must be a numerical value")

   if not isinstance(maxit, int):
       raise TypeError("maxit must be an integer")

   test = func(xl, *args) * func(xu, *args)
   if test > 0:
       raise ValueError("No sign change in the function between xl and xu")

   iter = 0
   xr = xl
   ea = 100

   while True:
       xrold = xr
       xr = (xl + xu) / 2
       iter += 1
       if xr != 0:
           ea = abs((xr - xrold) / xr) * 100

       if test < 0:
           xu = xr
       elif test > 0:
           xl = xr
       else:
           ea = 0

       if ea <= es or iter >= maxit:
           break

   root = xr
   fx = func(xr, *args)
   return root, fx, ea, iter

if __name__ == "__main__":
    # Example usage
    def f(x):
        return x**3 - 6*x**2 + 11*x - 6

    xl = 1
    xu = 3
    es = 0.01
    maxit = 100

    root, fx, ea, iter = bisect(f, xl, xu, es, maxit)
    print("Root:", root)
    print("Function value at root:", fx)
    print("Approximate relative error:", ea, "%")
    print("Iterations:", iter)