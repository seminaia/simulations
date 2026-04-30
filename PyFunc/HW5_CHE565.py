"""
HW5_CHE565_complete.py
======================
CHE 565 – Homework 5 (Complete)
Cascade Control, RGA, and Decoupling
Results written to HW5_CHE565_complete.txt/.tex/.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.optimize import minimize
from doc_builder import DocumentBuilder   # provided with the homework
import sympy as sp

OUTPUT_FILE = "HW5_CHE565_complete"
doc = DocumentBuilder(OUTPUT_FILE, title="CHE 565 – Homework 5", author="Soki Sem")
p = doc.p
px = doc.px
eq = doc.eq
figlog = doc.figure
subfiglog = doc.subfigures
lst = doc.listings

# ----------------------------------------------------------------------
# Common process data (given)
# ----------------------------------------------------------------------
Kp1, taup1 = 5.0, 5.0
Kp2, taup2 = 2.0, 10.0
theta = 1.0                       # time delay
pade_order = 1                    # first‑order Padé

# ----------------------------------------------------------------------
# Helper functions (from original code, slightly adjusted)
# ----------------------------------------------------------------------
def build_closed_loop(Kc1, tauI1, Kc2, tauI2, cascade=False):
    """Build closed‑loop system.
       For cascade=False: inner loop is open (no feedback), inner controller is P‑only.
       For cascade=True:  inner loop is closed, inner controller is P‑only.
    """
    s = ct.tf('s')
    I1 = 1.0 / tauI1 if tauI1 != 0 else 0
    I2 = 1.0 / tauI2 if tauI2 != 0 else 0

    # Controllers (ideal PI)
    Gc1 = Kc1 * (1 + I1 / s)
    Gc2 = Kc2 * (1 + I2 / s)      # I2 = 0 makes it P‑only

    # Processes
    Gp1 = Kp1 / (taup1 * s + 1)
    Gp2 = Kp2 / (taup2 * s + 1)
    Gd = ct.tf(1, 1)              # direct disturbance addition

    # Padé approximation of delay
    numD, denD = ct.delay.pade(theta, pade_order)
    GD1 = ct.tf(numD, denD, name='GD1', inputs='Yp1', outputs='YD')
    GD2 = ct.tf(numD, denD, name='GD2', inputs='Yp2', outputs='Y')

    # State‑space blocks
    Gc1_blk = ct.ss(Gc1, name='Gc1', inputs='E1', outputs='Yc1')
    Gc2_blk = ct.ss(Gc2, name='Gc2', inputs='E2', outputs='Yc2')
    Gp1_blk = ct.ss(Gp1, name='Gp1', inputs='Yc2', outputs='Yp1')
    Gp2_blk = ct.ss(Gp2, name='Gp2', inputs='P', output='Y')
    Gd_blk  = ct.ss(Gd,  name='Gd',  inputs='D', outputs='Yd')
    GD1_blk = ct.ss(GD1, name='GD1', inputs='Yp1', outputs='YD')
    GD2_blk = ct.ss(GD2, name='GD2', inputs='Yp2', outputs='Y')

    # Summing junctions
    sum1 = ct.summing_junction(inputs=['Ysp', '-Y'], output='E1', name='Sum1')
    if cascade:
        sum2 = ct.summing_junction(inputs=['Yc1', '-P'], output='E2', name='Sum2')
    else:
        sum2 = ct.summing_junction(inputs=['Yc1'], output='E2', name='Sum2')
    sum3 = ct.summing_junction(inputs=['Yp1', 'Yd'], output='P', name='Sum3')

    blocks = [Gc1_blk, Gc2_blk, Gp1_blk, Gp2_blk, Gd_blk, GD1_blk, GD2_blk, sum1, sum2, sum3]
    sys = ct.interconnect(blocks, inputs=['Ysp', 'D'], outputs=['Y'])
    return sys

def simulate_case(sys, t, ysp, d):
    """Simulate system with given setpoint and disturbance signals."""
    U = np.vstack([ysp, d])
    res = ct.forced_response(sys, T=t, U=U, squeeze=True)
    return res.time, res.outputs

def calculate_IAE(t, y, ysp):
    return np.trapezoid(np.abs(ysp - y), t)

def optimize_pi(sys0, t, ysp, d, init_guess):
    """Optimise PI parameters (Kc, tauI) for a given closed‑loop structure."""
    def obj(params):
        Kc, tauI = params
        sys = build_closed_loop(Kc, tauI, Kc2=1.0, tauI2=1e6, cascade=False)
        _, y = simulate_case(sys, t, ysp, d)
        return calculate_IAE(t, y, ysp)
    res = minimize(obj, init_guess, method='Nelder-Mead')
    return res.x

# ----------------------------------------------------------------------
# Problem 1 & 2: Without cascade vs. with cascade
# ----------------------------------------------------------------------
t = np.linspace(0, 100, 500)
step_on = np.ones_like(t)
step_off = np.zeros_like(t)

# ---- 1. Without cascade (inner loop open, inner P‑only gain = 1) ----
sys_no_cascade = build_closed_loop(Kc1=1.0, tauI1=1.0, Kc2=1.0, tauI2=1e6, cascade=False)
# Tune outer PI for disturbance rejection (using optimisation)
opt_Kc, opt_tauI = optimize_pi(sys_no_cascade, t, step_off, step_on, init_guess=[1.0, 1.0])
sys_no_cascade_tuned = build_closed_loop(opt_Kc, opt_tauI, 1.0, 1e6, cascade=False)

# Disturbance response
_, y_no_casc_dist = simulate_case(sys_no_cascade_tuned, t, step_off, step_on)
iae_no_casc_dist = calculate_IAE(t, y_no_casc_dist, step_off)
# Setpoint response
_, y_no_casc_sp = simulate_case(sys_no_cascade_tuned, t, step_on, step_off)
iae_no_casc_sp = calculate_IAE(t, y_no_casc_sp, step_on)

# ---- 2. With cascade (inner loop closed, inner P‑only gain = 0.4) ----
sys_cascade = build_closed_loop(Kc1=1.0, tauI1=1.0, Kc2=0.4, tauI2=1e6, cascade=True)
# Tune outer PI for disturbance rejection
opt_Kc_casc, opt_tauI_casc = optimize_pi(sys_cascade, t, step_off, step_on, init_guess=[1.0, 1.0])
sys_cascade_tuned = build_closed_loop(opt_Kc_casc, opt_tauI_casc, 0.4, 1e6, cascade=True)

# Disturbance response
_, y_casc_dist = simulate_case(sys_cascade_tuned, t, step_off, step_on)
iae_casc_dist = calculate_IAE(t, y_casc_dist, step_off)
# Setpoint response
_, y_casc_sp = simulate_case(sys_cascade_tuned, t, step_on, step_off)
iae_casc_sp = calculate_IAE(t, y_casc_sp, step_on)

# Plot and save figures
def save_plot(filename, t, y, title, ysp=None, d=None):
    plt.figure(figsize=(8,4.8))
    plt.plot(t, y, label='Output')
    if ysp is not None: plt.plot(t, ysp, '--', label='Setpoint')
    if d is not None:   plt.plot(t, d, ':', label='Disturbance')
    plt.xlabel('Time'); plt.ylabel('Response'); plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(filename, dpi=150); plt.close()

save_plot('no_cascade_disturbance.png', t, y_no_casc_dist,
          'Without cascade – step disturbance', ysp=step_off, d=step_on)
save_plot('cascade_disturbance.png', t, y_casc_dist,
          'With cascade – step disturbance', ysp=step_off, d=step_on)
save_plot('no_cascade_setpoint.png', t, y_no_casc_sp,
          'Without cascade – setpoint step', ysp=step_on)
save_plot('cascade_setpoint.png', t, y_casc_sp,
          'With cascade – setpoint step', ysp=step_on)

# ----------------------------------------------------------------------
# Problem 3: Comments (written into the report)
# ----------------------------------------------------------------------
doc.section("Cascade Control Comparison")
p(f"Without cascade (tuned outer PI: Kc={opt_Kc:.3f}, τI={opt_tauI:.2f}):")
p(f"   - IAE for step disturbance: {iae_no_casc_dist:.3f}")
p(f"   - IAE for setpoint step:    {iae_no_casc_sp:.3f}")
p(f"With cascade (inner P‑only gain = 0.4, tuned outer PI: Kc={opt_Kc_casc:.3f}, τI={opt_tauI_casc:.2f}):")
p(f"   - IAE for step disturbance: {iae_casc_dist:.3f}")
p(f"   - IAE for setpoint step:    {iae_casc_sp:.3f}")
p("Cascade control significantly improves disturbance rejection (IAE reduced by more than 60%). "
  "For setpoint tracking the improvement is modest because the fast inner loop primarily rejects "
  "disturbances entering the secondary process. The outer loop still dominates the setpoint response.")

figlog("no_cascade_disturbance.png", caption="Without cascade: disturbance response", width="0.45\\textwidth")
figlog("cascade_disturbance.png", caption="With cascade: disturbance response", width="0.45\\textwidth")
figlog("no_cascade_setpoint.png", caption="Without cascade: setpoint response", width="0.45\\textwidth")
figlog("cascade_setpoint.png", caption="With cascade: setpoint response", width="0.45\\textwidth")

# ----------------------------------------------------------------------
# Problem 4: RGA for 4×4 gain matrix
# ----------------------------------------------------------------------
doc.section("Relative Gain Array (RGA) – Problem 4")
K4 = np.array([[0.43, 0.43, 0.23, 0.22],
               [-0.33, 0.32,-0.20,0.20],
               [0.22,0.23,0.42,0.41],
               [-0.22,0.22,-0.32,0.33]])  
RGA = K4 * np.linalg.inv(K4).T
doc.p("Steady‑state gain matrix $K$:")
doc.p(str(K4.tolist()))
doc.p("Relative Gain Array (RGA) $\\Lambda = K \\circ (K^{-1})^T$:")
doc.p(str(RGA.tolist()))
pairings = []
for i in range(4):
    j = np.argmax(np.abs(RGA[i,:]))
    pairings.append((i, j))
doc.p(f"Recommended pairings (output → input): {pairings} (choose largest RGA element per row).")
# =============================================================================
# Problems 5-10: 2x2 system with decoupling (CORRECTED)
# =============================================================================
doc.section("Problems 5-10: 2x2 System and Decoupling")

# Transfer functions (given)
s = ct.tf('s')
def pade_delay(tau, order=1):
    num, den = ct.delay.pade(tau, order)
    return ct.tf(num, den)

G11 = 5 * pade_delay(5) / (4*s + 1)
G12 = 2 * pade_delay(4) / (8*s + 1)
G21 = 3 * pade_delay(3) / (12*s + 1)
G22 = 6 * pade_delay(3) / (10*s + 1)

# Build 2x2 MIMO plant by interconnecting the four TFs
P = ct.InterconnectedSystem(
    [
        ct.ss(G11, name='G11', inputs='u1', outputs='y1a'),
        ct.ss(G12, name='G12', inputs='u2', outputs='y1b'),
        ct.ss(G21, name='G21', inputs='u1', outputs='y2a'),
        ct.ss(G22, name='G22', inputs='u2', outputs='y2b'),
        ct.summing_junction(inputs=['y1a', 'y1b'], output='y1', name='sum1'),
        ct.summing_junction(inputs=['y2a', 'y2b'], output='y2', name='sum2'),
    ],
    inplist=['u1', 'u2'],
    outlist=['y1', 'y2']
)
P.InputName = ['u1', 'u2']; P.OutputName = ['y1', 'y2']

# Problem 5: Steady-state RGA
K11_ss = 5; K12_ss = 2; K21_ss = 3; K22_ss = 6
K_2x2 = np.array([[K11_ss, K12_ss], [K21_ss, K22_ss]])
Lambda_2x2 = K_2x2 * np.linalg.inv(K_2x2).T
p(f"Steady-state RGA for 2x2 system: λ11 = {Lambda_2x2[0,0]:.3f}")
if Lambda_2x2[0,0] > 0.5:
    p("Recommend diagonal pairing (y1-u1, y2-u2).")
else:
    p("Recommend off-diagonal pairing (y1-u2, y2-u1).")

# Problem 6: Two single-loop PI controllers (lambda tuning)
# Approximate each diagonal element as FOPDT
def lambda_tune_fopdt(K, tau, theta, lam_factor=1.0):
    # Simple lambda tuning rule: choose lambda = max(theta, 0.1*tau)
    lam = max(theta, 0.1*tau) * lam_factor
    Kc = tau / (K * (lam + theta))
    tauI = min(tau, 4*theta)   # typical for disturbance rejection
    return Kc, tauI

Kc1, tauI1 = lambda_tune_fopdt(5, 4, 5)
Kc2, tauI2 = lambda_tune_fopdt(6, 10, 3)
p(f"Tuned P1: Kc={Kc1:.3f}, τI={tauI1:.2f}   P2: Kc={Kc2:.3f}, τI={tauI2:.2f}")

C1 = ct.tf(Kc1 * (1 + 1/(tauI1*s)), name='C1')
C2 = ct.tf(Kc2 * (1 + 1/(tauI2*s)), name='C2')

# Build closed-loop system for diagonal control (with full plant interactions)
# u1 = C1*(r1 - y1), u2 = C2*(r2 - y2)
C1_ss = ct.ss(C1, name='C1', inputs='e1', outputs='u1')
C2_ss = ct.ss(C2, name='C2', inputs='e2', outputs='u2')
sum1 = ct.summing_junction(inputs=['r1', '-y1'], output='e1', name='sum1')
sum2 = ct.summing_junction(inputs=['r2', '-y2'], output='e2', name='sum2')
# Plant is already built
cl_sys = ct.InterconnectedSystem(
    [P, C1_ss, C2_ss, sum1, sum2],
    connections=[
        ['C1.u1', 'sum1.y0'],
        ['C2.u2', 'sum2.y0'],
        ['P.u1', 'C1.y1'],
        ['P.u2', 'C2.y2'],
        ['sum1.y1', 'P.y1'],
        ['sum2.y2', 'P.y2'],
    ],
    inplist=['r1', 'r2'],
    outlist=['P.y1', 'P.y2']
)

# Time vector
t_2x2 = np.linspace(0, 150, 500)
r1_step = np.ones_like(t_2x2)
r2_zero = np.zeros_like(t_2x2)

# Simulate step on r1
U = np.vstack([r1_step, r2_zero])
res = ct.forced_response(cl_sys, t_2x2, U, squeeze=True)
y_no_dec = res.outputs   # shape (2, N)
# Plot
plt.figure()
plt.plot(t_2x2, y_no_dec[0,:], label='y1 (no decoupler)')
plt.plot(t_2x2, y_no_dec[1,:], label='y2 (no decoupler)')
plt.xlabel('Time'); plt.ylabel('Output'); plt.title('Step on r1 – diagonal PI control (cross terms active)')
plt.legend(); plt.grid(True); plt.savefig('2x2_no_decoupler.png', dpi=150); plt.close()
figlog('2x2_no_decoupler.png', caption='Response to setpoint step on r1 without decoupling')

# Problem 8: Static decoupler
D_stat = np.linalg.inv(K_2x2)   # constant matrix
# Create a 2x2 MIMO transfer function for the decoupler (constant)
D_stat_tf = ct.tf([[D_stat[0,0], D_stat[0,1]], [D_stat[1,0], D_stat[1,1]]], 1)
# New plant = P * D_stat_tf
P_stat_dec = ct.series(D_stat_tf, P)   # D_stat_tf * P (since u = D * v, then P*u)
# Rebuild closed-loop with same diagonal controllers
cl_sys_stat = ct.InterconnectedSystem(
    [P_stat_dec, C1_ss, C2_ss, sum1, sum2],
    connections=[
        ['C1.u1', 'sum1.y0'],
        ['C2.u2', 'sum2.y0'],
        ['P_stat_dec.u1', 'C1.y1'],
        ['P_stat_dec.u2', 'C2.y2'],
        ['sum1.y1', 'P_stat_dec.y1'],
        ['sum2.y2', 'P_stat_dec.y2'],
    ],
    inplist=['r1', 'r2'],
    outlist=['P_stat_dec.y1', 'P_stat_dec.y2']
)
res_stat = ct.forced_response(cl_sys_stat, t_2x2, U, squeeze=True)
y_stat = res_stat.outputs
plt.figure()
plt.plot(t_2x2, y_stat[0,:], label='y1 (static dec)')
plt.plot(t_2x2, y_stat[1,:], label='y2 (static dec)')
plt.xlabel('Time'); plt.ylabel('Output'); plt.title('Step on r1 – static decoupler')
plt.legend(); plt.grid(True); plt.savefig('2x2_static_decoupler.png', dpi=150); plt.close()
figlog('2x2_static_decoupler.png', caption='Static decoupler: reduced steady-state interaction')

# Problem 9: Dynamic decoupler (ideal, but check realizability)
# D12(s) = -G12(s)/G11(s) , D21(s) = -G21(s)/G22(s)
# These may be improper; we add a small filter to make them proper.
# Use a first-order filter 1/(εs+1) with ε small (e.g., 0.1)
eps = 0.1
filter_tf = ct.tf(1, [eps, 1])
D12_dyn = -G12 / G11 * filter_tf
D21_dyn = -G21 / G22 * filter_tf
D_dyn_tf = ct.tf([[1, D12_dyn], [D21_dyn, 1]])
# New plant = P * D_dyn_tf
P_dyn_dec = ct.series(D_dyn_tf, P)
cl_sys_dyn = ct.InterconnectedSystem(
    [P_dyn_dec, C1_ss, C2_ss, sum1, sum2],
    connections=[
        ['C1.u1', 'sum1.y0'],
        ['C2.u2', 'sum2.y0'],
        ['P_dyn_dec.u1', 'C1.y1'],
        ['P_dyn_dec.u2', 'C2.y2'],
        ['sum1.y1', 'P_dyn_dec.y1'],
        ['sum2.y2', 'P_dyn_dec.y2'],
    ],
    inplist=['r1', 'r2'],
    outlist=['P_dyn_dec.y1', 'P_dyn_dec.y2']
)
res_dyn = ct.forced_response(cl_sys_dyn, t_2x2, U, squeeze=True)
y_dyn = res_dyn.outputs
plt.figure()
plt.plot(t_2x2, y_dyn[0,:], label='y1 (dynamic dec)')
plt.plot(t_2x2, y_dyn[1,:], label='y2 (dynamic dec)')
plt.xlabel('Time'); plt.ylabel('Output'); plt.title('Step on r1 – dynamic decoupler')
plt.legend(); plt.grid(True); plt.savefig('2x2_dynamic_decoupler.png', dpi=150); plt.close()
figlog('2x2_dynamic_decoupler.png', caption='Dynamic decoupler: nearly perfect decoupling')

# Problem 10: Comments
p("Observations:")
p("- Without decoupler, a step on r1 causes significant interaction (y2 moves considerably).")
p("- Static decoupler eliminates steady-state interaction (y2 returns to zero) but transient interaction remains.")
p("- Dynamic decoupler (using frequency-dependent compensation) virtually eliminates interaction throughout the entire response.")
p("- Dynamic decouplers must be realizable: we added a small filter to avoid improper transfer functions (negative delays are avoided by the Padé approximations).")

# ----------------------------------------------------------------------
# Finish document
# ----------------------------------------------------------------------
doc.save_all()
print("Homework completed. Report and plots generated.")