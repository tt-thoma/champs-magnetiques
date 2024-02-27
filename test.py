import numpy as np
from scipy.constants import mu_0, epsilon_0
from scipy.ndimage import convolve
from scipy import integrate


E = np.array([E_x, E_y, E_z])

dt = 0.01
curl_E = np.array(np.gradient(E))[np.array([1,2,0])]-np.array(np.gradient(E))[np.array([2,0,1])]
dBdt = -curl_E
B = integrate.simps(dBdt, dx=dt)


