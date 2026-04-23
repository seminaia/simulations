import control as ct
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Problem data
# ---------------------------
Kp = 1          # process gain
taup = 5        # process time constant
theta = 1       # dead time
pade_order = 3  # for delay approximation

# ---------------------------
# Lambda tuning rules for PI
# Gp(s) = Kp * exp(-theta s) / (taup s + 1)
# lambda = max(taup/3, theta)
# Kc = taup / [Kp * (lambda + theta)]
# tauI = taup
# Simulink uses I = 1/tauI
# ---------------------------
lam = max(taup / 3, theta)
Kc_nom = taup / (Kp * (lam + theta))
tauI_nom = taup
I_nom = 1 / tauI_nom

print("Lambda tuning results")
print(f"lambda = {lam:.4f}")
print(f"Kc = {Kc_nom:.4f}")
print(f"tauI = {tauI_nom:.4f}")
print(f"I = 1/tauI = {I_nom:.4f}")

# ---------------------------
# Time variable for transfer functions
# ---------------------------
s = ct.tf('s')

# ---------------------------
# Build closed-loop system
# Disturbance is added AFTER controller:
# P = Yc + d
# Y = delay * [ Kp/(taup s + 1) ] * P
# E = Ysp - Y
# Yc = Gc * E
# ---------------------------
def build_closed_loop(Kc, tauI):
    I = 1 / tauI

    # PI controller
    Gc = Kc * (1 + I / s)

    # plant without delay
    Gp_nodelay = Kp / (taup * s + 1)

    # Pade approximation of delay exp(-theta s)
    numD, denD = ct.pade(theta, pade_order)
    Gdelay = ct.tf(numD, denD)

    # Named blocks
    Gc_blk = ct.tf(Gc, name='Gc', inputs='E', outputs='Yc')
    Gp_blk = ct.tf(Gp_nodelay, name='Gp', inputs='P', outputs='Yp')
    Gd_blk = ct.tf(1, name='Gd', inputs='d', outputs='Yd')     # direct disturbance addition
    GD_blk = ct.tf(Gdelay, name='GD', inputs='Yp', outputs='Y')

    # summing junctions
    sum1 = ct.summing_junction(inputs=['Ysp', '-Y'], output='E', name='Sum1')
    sum2 = ct.summing_junction(inputs=['Yc', 'Yd'], output='P', name='Sum2')

    # interconnected system: inputs are setpoint Ysp and disturbance d
    sys = ct.interconnect(
        [Gc_blk, Gp_blk, Gd_blk, GD_blk, sum1, sum2],
        inplist=['Ysp', 'd'],
        outlist=['Y']
    )
    return sys

# ---------------------------
# Simulation helper
# ---------------------------
def simulate_case(sys, t, ysp_input, d_input):
    U = np.vstack([ysp_input, d_input])   # 2 inputs: Ysp and d
    resp = ct.forced_response(sys, T=t, U=U)
    return resp.time, resp.outputs

# ---------------------------
# Plot helper
# ---------------------------
def plot_case(t, y, title, ysp=None, d=None):
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, y, label='Output y(t)', linewidth=2)

    if ysp is not None:
        plt.plot(t, ysp, '--', label='Setpoint', linewidth=1.5)

    if d is not None:
        plt.plot(t, d, ':', label='Disturbance input', linewidth=1.5)

    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# Time vector and step signals
# ---------------------------
t = np.linspace(0, 50, 1000)

step_on = np.ones_like(t)
step_off = np.zeros_like(t)

# ---------------------------
# Nominal controller
# ---------------------------
sys_nom = build_closed_loop(Kc_nom, tauI_nom)
doc.section("Problem 1")
doc.subsection("System Definition")

m(r"G_p(s) = \frac{e^{-s}}{5s + 1}")

w("The given system is a first-order plus dead-time (FOPDT) model.")

m(r"K_p = 1, \quad \tau_p = 5, \quad \theta = 1")

doc.subsection("Controller Design using \\lambda-Rules")

m(r"\lambda = \max\left(\frac{\tau_p}{3}, \theta \right)")

a(
    r"\lambda &= \max\left(\frac{5}{3}, 1\right)",
    r"&= \frac{5}{3} \approx 1.667"
)

m(r"K_c = \frac{\tau_p}{K_p(\lambda + \theta)}")

a(
    r"K_c &= \frac{5}{1(1.667 + 1)}",
    r"&= \frac{5}{2.667}",
    r"&\approx 1.875"
)

m(r"\tau_I = \tau_p")

a(
    r"\tau_I &= 5"
)

m(r"I = \frac{1}{\tau_I}")

a(
    r"I &= \frac{1}{5}",
    r"&= 0.2"
)
doc.subsection("Controller Transfer Function")

m(r"G_c(s) = K_c \left(1 + \frac{1}{\tau_I s} \right)")

a(
    r"G_c(s) &= 1.875 \left(1 + \frac{1}{5s} \right)",
    r"&= 1.875 + \frac{0.375}{s}"
)

doc.subsection("Closed-Loop System")

w("The control system is arranged in a standard feedback configuration.")

m(r"E = Y_{sp} - Y")

m(r"Y_c = G_c(s)E")

w("The disturbance is added after the controller output.")

m(r"P = Y_c + d")

m(r"Y = G_p(s) P")
doc.subsection("Simulation Cases")

w("The following simulations were performed:")

w("1. Step change in setpoint (magnitude 1), no disturbance")
w("2. Step change in disturbance (magnitude 1), no setpoint change")
w("3. Double controller gain")
w("4. Half controller gain")
w("5. Increased and decreased integral time constant")

# 1) Setpoint step, no disturbance
t1, y1 = simulate_case(sys_nom, t, step_on, step_off)
plot_case(t1, y1, f'Nominal PI: Setpoint Step, No Disturbance\nKc={Kc_nom:.3f}, tauI={tauI_nom:.3f}',
          ysp=step_on, d=step_off)
doc.subsection("Setpoint Response")

w("A unit step change in setpoint was applied with no disturbance.")

m(r"Y_{sp}(t) = 1, \quad d(t) = 0")
# 2) Disturbance step, no setpoint change
t2, y2 = simulate_case(sys_nom, t, step_off, step_on)
plot_case(t2, y2, f'Nominal PI: Disturbance Step, No Setpoint Change\nKc={Kc_nom:.3f}, tauI={tauI_nom:.3f}',
          ysp=step_off, d=step_on)
doc.subsection("Disturbance Response")

w("A unit step disturbance was applied with no setpoint change.")

m(r"Y_{sp}(t) = 0, \quad d(t) = 1")

# ---------------------------
# 3) Double controller gain
# ---------------------------
Kc_high = 2 * Kc_nom
sys_Kc_high = build_closed_loop(Kc_high, tauI_nom)

t3, y3 = simulate_case(sys_Kc_high, t, step_on, step_off)
plot_case(t3, y3, f'Double Kc: Setpoint Step, No Disturbance\nKc={Kc_high:.3f}, tauI={tauI_nom:.3f}',
          ysp=step_on, d=step_off)
doc.subsection("Effect of Controller Gain")

w("The controller gain was varied to observe its effect on system response.")

m(r"K_c = 2K_c^{\text{nominal}}")

m(r"K_c = 0.5K_c^{\text{nominal}}")

t4, y4 = simulate_case(sys_Kc_high, t, step_off, step_on)
plot_case(t4, y4, f'Double Kc: Disturbance Step, No Setpoint Change\nKc={Kc_high:.3f}, tauI={tauI_nom:.3f}',
          ysp=step_off, d=step_on)

# ---------------------------
# 4) Half controller gain
# ---------------------------
Kc_low = 0.5 * Kc_nom
sys_Kc_low = build_closed_loop(Kc_low, tauI_nom)

t5, y5 = simulate_case(sys_Kc_low, t, step_on, step_off)
plot_case(t5, y5, f'Half Kc: Setpoint Step, No Disturbance\nKc={Kc_low:.3f}, tauI={tauI_nom:.3f}',
          ysp=step_on, d=step_off)

t6, y6 = simulate_case(sys_Kc_low, t, step_off, step_on)
plot_case(t6, y6, f'Half Kc: Disturbance Step, No Setpoint Change\nKc={Kc_low:.3f}, tauI={tauI_nom:.3f}',
          ysp=step_off, d=step_on)

# ---------------------------
# 5) Change tauI
#    increase tauI and decrease tauI
# ---------------------------
tauI_big = 2 * tauI_nom
tauI_small = 0.5 * tauI_nom

sys_tauI_big = build_closed_loop(Kc_nom, tauI_big)
sys_tauI_small = build_closed_loop(Kc_nom, tauI_small)

# Increased tauI
t7, y7 = simulate_case(sys_tauI_big, t, step_on, step_off)
plot_case(t7, y7, f'Increased tauI: Setpoint Step, No Disturbance\nKc={Kc_nom:.3f}, tauI={tauI_big:.3f}',
          ysp=step_on, d=step_off)

t8, y8 = simulate_case(sys_tauI_big, t, step_off, step_on)
plot_case(t8, y8, f'Increased tauI: Disturbance Step, No Setpoint Change\nKc={Kc_nom:.3f}, tauI={tauI_big:.3f}',
          ysp=step_off, d=step_on)

# Decreased tauI
t9, y9 = simulate_case(sys_tauI_small, t, step_on, step_off)
plot_case(t9, y9, f'Decreased tauI: Setpoint Step, No Disturbance\nKc={Kc_nom:.3f}, tauI={tauI_small:.3f}',
          ysp=step_on, d=step_off)

t10, y10 = simulate_case(sys_tauI_small, t, step_off, step_on)
plot_case(t10, y10, f'Decreased tauI: Disturbance Step, No Setpoint Change\nKc={Kc_nom:.3f}, tauI={tauI_small:.3f}',
doc.subsection("Effect of Integral Time Constant")

w("The integral time constant was varied to study its impact.")

m(r"\tau_I = 2\tau_I^{\text{nominal}}")

m(r"\tau_I = 0.5\tau_I^{\text{nominal}}")
doc.subsection("Discussion")

w("Increasing the controller gain results in a faster response but increases overshoot and oscillations.")

w("Decreasing the controller gain results in a slower but more stable response.")

w("Increasing the integral time constant reduces the strength of integral action, slowing offset correction.")

w("Decreasing the integral time constant strengthens integral action, improving steady-state accuracy but potentially causing oscillations.")
doc.subsection("Final Controller Parameters")

a(
    r"K_c &= 1.875",
    r"\tau_I &= 5",
    r"I &= 0.2"
)
# ---------------------------
# Optional comparison plots
# ---------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(t1, y1, label='Nominal')
plt.plot(t3, y3, label='Double Kc')
plt.plot(t5, y5, label='Half Kc')
plt.plot(t, step_on, '--', label='Setpoint')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Effect of Kc on Setpoint-Tracking Response')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4.5))
plt.plot(t2, y2, label='Nominal')
plt.plot(t4, y4, label='Double Kc')
plt.plot(t6, y6, label='Half Kc')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Effect of Kc on Disturbance-Rejection Response')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4.5))
plt.plot(t1, y1, label='Nominal tauI')
plt.plot(t7, y7, label='Increased tauI')
plt.plot(t9, y9, label='Decreased tauI')
plt.plot(t, step_on, '--', label='Setpoint')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Effect of tauI on Setpoint-Tracking Response')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4.5))
plt.plot(t2, y2, label='Nominal tauI')
plt.plot(t8, y8, label='Increased tauI')
plt.plot(t10, y10, label='Decreased tauI')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Effect of tauI on Disturbance-Rejection Response')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
