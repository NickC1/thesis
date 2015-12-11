import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate
from sklearn import metrics
sns.set()
from sklearn import neighbors

sns.set_context('paper',font_scale=2)
sns.set_style('ticks')

def lorenz(sz=10000,noise=0):
    x0 = [1, 1, 1]  # starting vector
    t = np.linspace(0, 100, sz)  # one thousand time steps
    X = integrate.odeint(lorentz_deriv, x0, t)

    return X


def lorentz_deriv((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
    """Defining lorenz equation to call above"""
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


X = lorenz(sz=10000)[::5,0]
plt.plot(X)
plt.xlabel("Time",size=25)
plt.ylabel(r'$X$',size=25)
sns.despine()
