import control as ct 
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import sympy as sp
m, c, k = 1, 0.1, 2

# Create a linear system
# m*q1_ddot = -2k*q1 + k*q2 - c*q1_dot
# m*q2_ddot = k*q1 - 2k*q2 - c*q2_dot + k*u

# state function
# x1 = q1
# x2 = q2
# x1_dot = q1_dot = x3
# x2_dot = q2_dot = x4
# x3_dot = -2k/m*x1 + k/m*x2 - c/m*x3
# x4_dot = k/m*x1 - 2k/m*x2 - c/m*x4 + k/m*u
# A =
A = np.array([
    [0, 0, 1, 0], # [x1, x2, x3, x4]
    [0, 0, 0, 1],
    [-2*k/m, k/m, -c/m, 0],
    [k/m, -2*k/m, 0, -c/m]
])

B = np.array([[0], [0], [0], [k/m]]) # Input function u, which only affects x4_dot or the second mass
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # Output Function, we want to observe x1 and x2
D = 0

sys = ct.ss(A, B, C, D, outputs=['q1', 'q2'], name="coupled spring mass")

response = ct.initial_response(sys, X0=[1, 0, 0, 0])
t = response.time
x = response.states
plt.plot(t, x[0], 'b', t, x[1], 'r')
plt.legend(['$x_1$', '$x_2$'])
plt.xlim(0, 50)
plt.ylabel('States')
plt.xlabel('Time [s]')
plt.title("Initial response from $x_1 = 1$, $x_2 = 0$")
plt.show()