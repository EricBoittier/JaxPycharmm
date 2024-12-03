# Statistics
# Miscellaneous
import ase.units as units
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acovf
import pint
from scipy.signal import find_peaks

ureg = pint.UnitRegistry()

# Constants
c = units._c * ureg("m/s")
c = c.to("cm/s").magnitude
hbar = 5.29 * 10**-12 * ureg("cm^-1 s")
hbar = hbar.magnitude
# beta = 1 / (units.kB * 300 * ureg("K"))
# beta = beta.to("cm^-1").magnitude
# Quantum correction factor qcf
Kb = 3.1668114e-6  # Boltzmann constant in atomic units (Hartree/K)
beta = 1.0 / (Kb / float(300))  # atomic units

def autocorrelation_ft(series, timestep, verbose=False):
    """
    Compute the autocorrelation function of a time series using the Fourier
    """
    # Time for speed of light in vacuum to travel 1cm (0.01) in 1fs (1e15)
    jiffy = 0.01 / units._c * 1e12
    # Frequency range in cm^-1
    nframes = series.shape[0]
    print("Nframes: ", nframes)
    nfreq = int(nframes / 2) + 1
    freq = np.arange(nfreq) / float(nframes) / timestep * jiffy
    # Dipole-Dipole autocorrelation function
    acvx = acovf(series[:, 0], fft=True)
    acvy = acovf(series[:, 1], fft=True)
    acvz = acovf(series[:, 2], fft=True)
    acv = acvx + acvy + acvz
    print("ACV: ", acv.shape)
    acv = acv * np.blackman(nframes)
    # print("ACV: ", acv.shape)
    spectra = np.abs(np.fft.rfftn(acv))
    return freq, spectra

def intensity_correction(freq, spectra, volume):
    twopiomega = 2 * np.pi * freq
    exp_corr = (1 - np.exp(-beta * hbar * freq))
    three_h_c_v = 3 * hbar * c * volume
    spectra = spectra*(twopiomega * exp_corr) / three_h_c_v
    # scale the spectra
    # spectra = spectra / np.max(spectra)

    # scale the spectra so the integral is 1 between 0 and 4500 cm^-1
    spectra = spectra / np.trapz(spectra, freq)

    return freq, spectra

def read_dat_file(filename):
    """
    Read a .dat file and return the data as a numpy array.
    """
    dat = np.loadtxt(filename)
    return dat


def rolling_avg(freq, spectra, window=10):
    """
    Compute the rolling average of a data set.
    """
    freq = freq[window:]
    spectra = np.convolve(spectra, np.ones(window), 'valid') / window
    return freq, spectra[1:]

def assign_peaks(spectra, n_peaks=10, height=0.1):
    """
    Find peaks in the spectra.
    """
    distance = len(spectra) // n_peaks
    peaks = find_peaks(spectra, threshold=None, height=height, distance=distance)[0]
    return peaks