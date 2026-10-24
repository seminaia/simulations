import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==========================================
# 1. GEOMETRY & MATERIAL PROPERTIES (SI Units)
# ==========================================
class DCFC_Materials:
    A_cell = 10e-4  # 10 cm^2 active area (0.001 m^2)
    
    # Node 1: Induction Susceptor (Stainless Steel)
    L_sus = 2e-3    # 2 mm thick
    rho_sus = 8000  # kg/m^3
    Cp_sus = 500    # J/(kg K)
    k_sus = 20      # W/(m K)
    
    # Node 2: Molten Metal Anode (Liquid Tin, Sn)
    rho_anode = 7000    L_anode = 5e-3  # 5 mm depth

    Cp_anode = 250
    k_anode = 30
    
    # Node 3: YSZ Electrolyte
    L_ysz = 50e-6   # 50 um thick
    rho_ysz = 6000
    Cp_ysz = 500
    k_ysz = 2.5     # W/(m K)
    
    # Node 4: LCM Cathode (LaCaMnO3)
    L_cathode = 500e-6 # 500 um thick
    rho_cathode = 6500
    Cp_cathode = 500
    k_cathode = 3.0

mat = DCFC_Materials()

# Calculate Thermal Mass (C_th) and Thermal Resistances (R_th)
V_sus = mat.A_cell * mat.L_sus
C_th_sus = mat.rho_sus * V_sus * mat.Cp_sus
R_th_sus_anode = mat.L_sus / (mat.k_sus * mat.A_cell)

V_anode = mat.A_cell * mat.L_anode
C_th_anode = mat.rho_anode * V_anode * mat.Cp_anode
R_th_anode_ysz = (mat.L_anode/2 + mat.L_ysz/2) / (mat.k_anode * mat.A_cell)

V_ysz = mat.A_cell * mat.L_ysz
C_th_ysz = mat.rho_ysz * V_ysz * mat.Cp_ysz
R_th_ysz_cath = (mat.L_ysz/2 + mat.L_cathode/2) / (mat.k_ysz * mat.A_cell)

V_cath = mat.A_cell * mat.L_cathode
C_th_cath = mat.rho_cathode * V_cath * mat.Cp_cathode
# Convection resistance to ambient air (h*A approx 0.01 W/K for 10cm^2)
R_th_cath_amb = mat.L_cathode / (mat.k_cathode * mat.A_cell) + 0.01 

# ==========================================
# 2. ELECTROCHEMICAL MODELS (FIXED)
# ==========================================
def ysz_conductivity(T):
    """Ionic conductivity of 8YSZ (S/m) as a function of Temperature (K)"""
    # Corrected Arrhenius fit for 8YSZ (yields ~20 S/m at 800C)
    return 3.34e5 * np.exp(-10300 / T)

def nernst_voltage(T):
    """Nernst voltage for C + O2 -> CO2 (V)"""
    return 1.02 - 0.0001 * (T - 298)

# ==========================================
# 3. SYSTEM ODE DEFINITION
# ==========================================
def dcfc_system(t, y):
    T_sus, T_an, T_ysz, T_cath = y
    T_amb = 300.0  # Ambient temperature (K)
    
    # --- Induction Heating Control ---
    T_target = 1073.0  # 800 °C
    P_ind_max = 850.0  # FIXED: Increased to 850W to overcome room heat loss
    
    if T_sus < T_target:
        P_ind = P_ind_max * min(1.0, (T_target - T_sus) / 100.0)
    else:
        P_ind = 0.0
        
    # --- Electrochemistry ---
    I_op = 0.0
    Q_joule = 0.0
    Q_act = 0.0
    
    if T_ysz > 973.0: # Turns on above 700 °C
        I_op = 2.0    # 2 Amps
        R_ysz = mat.L_ysz / (ysz_conductivity(T_ysz) * mat.A_cell)
        
        Q_joule = I_op**2 * R_ysz 
        Q_act = I_op * 0.1 # 0.1V activation loss
        Q_rev = I_op * (-0.05) 
        Q_joule += Q_rev

    # --- Heat Transfer Equations ---
    # FIXED: Lowered ambient loss coefficient to 0.01 (realistic natural convection)
    dT_sus_dt = (P_ind 
                 - (T_sus - T_an) / R_th_sus_anode 
                 - (T_sus - T_amb) * 0.01) / C_th_sus
                 
    dT_an_dt = ((T_sus - T_an) / R_th_sus_anode 
                - (T_an - T_ysz) / R_th_anode_ysz) / C_th_anode
                
    dT_ysz_dt = ((T_an - T_ysz) / R_th_anode_ysz 
                 - (T_ysz - T_cath) / R_th_ysz_cath 
                 + Q_joule) / C_th_ysz
                 
    dT_cath_dt = ((T_ysz - T_cath) / R_th_ysz_cath 
                  - (T_cath - T_amb) / R_th_cath_amb 
                  + Q_act) / C_th_cath

    return [dT_sus_dt, dT_an_dt, dT_ysz_dt, dT_cath_dt]

# ==========================================
# 4. SIMULATION EXECUTION & PLOTTING
# ==========================================
T_init = [300, 300, 300, 300] 
t_span = (0, 1200) # 20 minutes
t_eval = np.linspace(t_span[0], t_span[1], 500)

print("Running DCFC Multiphysics Simulation...")
sol = solve_ivp(dcfc_system, t_span, T_init, method='BDF', t_eval=t_eval, rtol=1e-5, atol=1e-7)

T_sus = sol.y[0] - 273.15
T_anode = sol.y[1] - 273.15
T_ysz = sol.y[2] - 273.15
T_cath = sol.y[3] - 273.15
time_min = sol.t / 60.0

V_cell = np.zeros_like(time_min)
P_cell = np.zeros_like(time_min)

for i, T in enumerate(sol.y[2]):
    if T > 973.0:
        E_nernst = nernst_voltage(T)
        R_ysz = mat.L_ysz / (ysz_conductivity(T) * mat.A_cell)
        V_cell[i] = E_nernst - (2.0 * R_ysz) - 0.1 
        P_cell[i] = V_cell[i] * 2.0

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(time_min, T_sus, label='Susceptor (Induction)', linestyle='--')
ax1.plot(time_min, T_anode, label='Molten Metal Anode (Sn)')
ax1.plot(time_min, T_ysz, label='YSZ Electrolyte', linewidth=3, color='black')
ax1.plot(time_min, T_cath, label='LCM Cathode')
ax1.axhline(800, color='red', linestyle=':', label='Target Op. Temp (800°C)')
ax1.axhline(700, color='orange', linestyle=':', label='Min Op. Temp (700°C)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Transient Thermal Profile of DCFC with Inductive Heating')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(time_min, V_cell, label='Cell Voltage', color='blue')
ax2_twin = ax2.twinx()
ax2_twin.plot(time_min, P_cell, label='Electrical Power', color='red', linestyle='--')
ax2.set_ylabel('Voltage (V)', color='blue')
ax2_twin.set_ylabel('Power (W)', color='red')
ax2.set_xlabel('Time (minutes)')
ax2.set_title('Electrochemical Performance (Constant 2A Discharge)')
ax2.legend(loc='center left')
ax2_twin.legend(loc='center right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()