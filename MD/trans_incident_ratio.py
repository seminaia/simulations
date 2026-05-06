import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trans_incident_ratio(mu,rho,z):
    "I/I0=exp(-(mac)*rho*z)(mac=mu/rho)"
    mac = mu / rho
    I_I0 = np.exp(-(mac) * rho * z)
    return I_I0

def mac(k,lamda,Z): 
    "Mass attenuation coefficient in cm^2/g"
    mac = k * lamda**3 * Z**3
    return mac
    
mac1 =122.8  
rho = 11.34  # g/cm^3
mu1 = mac1 * rho 
mac2 = 84.13
mu2 = mac2 * rho
mac3 = 66.14 
mu3 = mac3 * rho
z = np.linspace(0, 0.002,10)  # cm
I_I01 = trans_incident_ratio(mu1, rho, z)
I_I02 = trans_incident_ratio(mu2, rho, z)
I_I03 = trans_incident_ratio(mu3, rho, z)

plt.figure(figsize=(10,12))
plt.plot(z, I_I01, label = r'Mo K-$\alpha$')
plt.plot(z, I_I02, label = r'Rh K-$\alpha$')
plt.plot(z, I_I03, label = r'Ag K-$\alpha$')
plt.legend()
plt.xlabel('Thickness z (cm)')
plt.ylabel('Transmitted to Incident Intensity Ratio I/I0')
plt.title('Transmitted to Incident Intensity Ratio vs Thickness')
plt.savefig('trans_incident_ratio.png')
plt.show()