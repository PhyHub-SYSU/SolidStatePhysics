import numpy as np
from matplotlib import pyplot as plt

from numpy import sin, cos, pi
from scipy.optimize import fsolve


def _Kronig_Penney(K, k, a, U0b):
    """
    Kronig-Penney model.
    """
    return U0b / (2*K) * sin(K * a) + cos(K * a) - cos(k * a)

@np.vectorize 
def E(k, n=1, a=1, U0b=4, **kwargs):
    """
    Get energy from k in Kronig-Penney model.
    ---
    Arguments:
        k: wavenumber, a number or an numpy.array.
        n: the n-th band, starting from 0.
        a: distance between atoms.
        U0b: the area of the potential wall.
        **kwargs: something else to pass to scipy.optimize.fsolve.
    Returns:
        The energies corresponding to k, in the unit of hbar^2/2m.
    """
    K = fsolve(_Kronig_Penney, (n + 1/2)*pi, args=(k, a, U0b))
    return K**2

def getBands(n=3, a=1, U0b=4, **kwargs):
    """
    Get n bands in Kronig-Penney model.
    ---
    Arguments:
        k: wavenumber, a number or an numpy.array.
        n: the n-th band, starting from 1.
        a: distance between atoms.
        U0b: the area of the potential wall.
        **kwargs: something else to pass to scipy.optimize.fsolve.
    Returns:
        bands: An array of the energies, in the unit of hbar^2/2m. 
            `bands[i]` is the i-th band.
    """
    k = np.linspace(-pi/a, pi/a, 100)
    bands = [E(k, i, a, U0b=4, **kwargs) for i in range(n)]
    return bands

def plotBands(n=3, a=1, U0b=4, 
    title=None, **kwargs):
    """
    Plot n bands in Kronig-Penney model.
    ---
    Arguments:
        k: wavenumber, a number or an numpy.array.
        n: the n-th band, starting from 1.
        a: distance between atoms.
        U0b: the area of the potential wall.
        title: the title of the ax. 
            If title == None then it is "Plotting of n bands".
        **kwargs: something else to pass to scipy.optimize.fsolve.
    Returns:
        (fig, ax)
    """
    if title == None:
        title = "Plotting of {} bands".format(n)
    k = np.linspace(-pi/a, pi/a, 100)
    bands = getBands(n, a, U0b, **kwargs)

    fig, ax = plt.subplots(figsize=(5, 15), dpi=200)

    for i, band in enumerate(bands):
        ax.plot(k, band, label="${}$-th band".format(i))
    
    ax.set_xlabel("$k$")
    ax.set_ylabel("$2m E/\hbar^2$")
    
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(True)

    ax.legend()

    return fig, ax

if __name__ == "__main__":
    plotBands()