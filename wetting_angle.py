import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def wetting_angle(theta):
    S = (2+np.cos(theta))*(1-np.cos(theta))**2 / 4
    return S

theta = np.linspace(0, np.pi, 100)
S = wetting_angle(theta)
plt.plot(theta*180/np.pi, S)
plt.xlabel('Wetting angle (degrees)')
plt.ylabel('Spreading parameter S')
plt.title('Spreading parameter vs Wetting angle')
plt.show()