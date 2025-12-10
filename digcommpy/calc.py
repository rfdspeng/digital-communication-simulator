# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:46:47 2022

General signal processing calculator functions

@author: Ryan Tsai
"""

# TODO ACLR calculator for single-carrier signals (right now it depends on SCS)
# TODO Compression calculator using binning (BAS? bin-average-smooth)
# TODO scale_psd - OFDM only right now
# TODO SDE windowing - symmetric or no?
# TODO propertly handle kwargs across functions


import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft

def aclr(x: np.ndarray, fs, bw, scs, en_plot=False):
    """
    Calculate ACLR

    Parameters
    ----------
    x: time-domain waveform (V)
    fs: sampling rate (MHz)
    bw: signal BW (MHz)
    scs: subcarrier spacing (kHz)
    en_plot: True/False
    
    Returns
    -------
    aclrm: low-side ACLR (dB)
    aclrp: high-side ACLR (dB)

    """

    rbw = scs/1000
    nrb = round(bw*5*15/scs)
    obw = nrb*12*scs/1000

    # Calculate signal frequencies
    sigl = -obw/2; sigh = obw/2-scs/1000
    sigf = [sigl,sigh]

    # Calculate high-side ACLR
    noisef = [bw-obw/2,bw+obw/2]
    aclrp = noise_dbc(x, fs, rbw, sigf, noisef, cfg={'en_plot': en_plot, 'title':"ACLR+"})

    # Calculate low-side ACLR
    noisef = [-bw-obw/2,-bw+obw/2]
    aclrm = noise_dbc(x, fs, rbw, sigf, noisef, cfg={'en_plot': en_plot,'title':"ACLR-"})
        
    return (aclrm, aclrp)

def comp_db(x: np.ndarray, y: np.ndarray, cfg: dict | None = None):
    """
    Calculate compression by using polynomial curve fitting

    Parameters
    ----------
    x: input signal (V)
    y: output signal (V)
    cfg: dictionary
        - 'polyorders': specify the polynomial terms, e.g. [1,3,5,7,9] for odd orders up to 9. By default, 1 through 9.
        - 'en_plot'
        - 'amam_fitting_title'
        - 'gain_title'

    Returns
    -------
    comp: estimated compression (dB)
    nlse: normalized least-squared fitting error (as a fraction/float of output power)

    """
    
    cfg = cfg if cfg else {}

    # Reshape to column vectors
    x = x.reshape(x.size, 1, copy=True)
    y = y.reshape(y.size, 1, copy=True)
    
    # Preprocess vectors
    x = x-x.mean()
    y = y-y.mean()
    x = np.abs(x)/np.abs(x).max()
    y = np.abs(y)/np.abs(y).max()
    
    # Generate kernel matrix, poly order 1 through 9
    ks = cfg.get("polyorders", range(1,10))
    X = np.empty((len(x), len(ks)))
    for idx, k in enumerate(ks):
        X[:, idx] = (x**k).squeeze()
    
    # Polynomial coefficient estimation
    # c = np.matmul(linalg.pinv(X), y)
    c = linalg.pinv(X) @ y
    
    # Generate the transfer curve using the estimated coefficients
    xe = np.linspace(0,1,2^16+1)
    ye = np.zeros_like(xe)
    ye2 = np.zeros_like(x)
    for k in ks:
        ye = ye+c[k-1]*xe**k
        ye2 = ye2+c[k-1]*x**k # To estimate LSE
    
    # Calculate the gain curve
    g = ye[1:]/xe[1:]

    # Calculate compression as (gain / gain at the beginning of the curve)
    comp = 20*np.log10(g[0]/g[-1])

    # Calculate the fitting error
    error = ye2-y
    error2 = (error*error).sum()
    power = (y*y).sum()
    nlse = 10*np.log10(error2/power).item() # least-squared error normalized to output power
    # nlse = nlse[0]
    
    if cfg.get("en_plot", False):
        fig = plt.figure()
        titlestr = cfg.get("amam_fitting_title", "AMAM")
        plt.plot(x, y, '.', label='Data')
        plt.plot(xe, ye, label='Fitted')
        plt.title(titlestr, {'fontsize':40})
        plt.xlabel("X (Normalized)", {'fontsize':30})
        plt.ylabel("Y (Normalized)", {'fontsize':30})
        plt.legend(loc="lower right", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        
        fig = plt.figure()
        titlestr = cfg.get("gain_title", "Gain")
        plt.plot(x, 20*np.log10(y/x), '.', label='Data')
        plt.plot(xe[1:], 20*np.log10(g), label='Fitted')
        plt.title(titlestr, {'fontsize':40})
        plt.xlabel("X (Normalized)", {'fontsize':30})
        plt.ylabel("Gain (dB)", {'fontsize':30})
        plt.legend(loc="lower left", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        
    return (comp, nlse)

def dbm2v(x, unit, zo=50):
    """
    Converts dBm to Volts and vice versa
    
    Parameters
    ----------
    x: value in dBm or Volts
    unit: 'V' or 'dBm'
    zo: impedance of the system in Ohms

    Returns
    -------
    P: power in dBm
    V: voltage in Volts

    """

    if unit == 'dBm':
        V = np.sqrt(10**(x/10)*zo*1e-3)
        return V
    elif unit == 'V':
        P = 10*np.log10(x**2/zo/1e-3)
        return P
    else:
        raise ValueError("unit must be either 'dBm' or 'V'")

def noise_dbc(x: np.ndarray, fs, rbw, sigf: list | tuple, noisef: list, cfg: dict | None = None):
    """
    Calculate noise power in a frequency band relative to signal power

    Parameters
    ----------
    x: time-domain signal (V)
    fs: sampling rate (MHz)
    rbw: desired resolution bandwidth for spectral density estimation (MHz)
    sigf: 2-element list or tuple defining the min and max signal frequencies (MHz)
    noisef: 2-element list or tuple defining the min and max noise frequencies (MHz)
    cfg: dictionary
        - 'wintype': 'kaiser', 'blackmanharris', 'hann'
        - 'en_plot'
        - 'title'

    Returns
    -------
    noise_dbc: noise power relative to signal power (dBc)

    """

    cfg = cfg if cfg else {
        "wintype": "kaiser",
        "beta":  25, # kaiser-specific kw arg
    }
    [p,f] = psd(x, fs, rbw, **cfg)
    
    sigfl, sigfh = sigf
    psig = p[(f >= sigfl) & (f <= sigfh)]
    fsig = f[(f >= sigfl) & (f <= sigfh)]
    
    noisefl, noisefh = noisef
    pnoise = p[(f >= noisefl) & (f <= noisefh)]
    fnoise = f[(f >= noisefl) & (f <= noisefh)]
    
    psig_sum = psig.sum()
    pnoise_sum = pnoise.sum()
    noise_dbc = 10*np.log10(pnoise_sum/psig_sum)
    
    if cfg.get("en_plot", False):
        fig = plt.figure()
        titlestr = cfg.get("title", "PSD")
        plt.plot(f, 10*np.log10(p), linewidth=2.5, label='Full PSD')
        plt.plot(fsig, 10*np.log10(psig), label='Signal')
        plt.plot(fnoise, 10*np.log10(pnoise), label='Noise')
        plt.title(titlestr, {'fontsize':40})
        plt.xlabel("Frequency (MHz)", {'fontsize':30})
        plt.ylabel("PSD (dBm)", {'fontsize':30})
        plt.legend(loc="lower center", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
    
    return noise_dbc

def papr(x: np.ndarray, p=99.99):
    """
    Calculate PAPR

    Parameters
    ----------
    x: time-domain signal (V)
    p: peak percentile in %. For example, if you wish to calculate 99.99% PAPR, set p = 99.99

    Returns
    -------
    papr: PAPR (dB)

    """

    # Calculate power (envelope^2) and sort from small to large. Works for both complex and real signals.
    env2 = np.sort((x*x.conj()).real)
    avg_power = env2.mean()
    peak_power = env2[np.ceil(env2.size*p/100).astype(int)]
    papr = 10*np.log10(peak_power/avg_power)
    
    return papr

def power_dbm(x: np.ndarray, zo=50):
    """
    Calculate power of a time-domain signal
    
    Parameters
    ----------
    x: signal in Volts
    zo: impedance of the system in Ohms
    
    Returns
    -------
    p_avg: average power in dBm
    p_peak: peak power in dBm
    
    """
    
    # Average power
    p_avg = 10*np.log10(rms(x)**2/zo/1e-3)
    
    # Peak power
    p_peak = 10*np.log10((np.abs(x).max())**2/zo/1e-3)

    return (p_avg, p_peak)
    
def psd(x: np.ndarray, fs, rbw, **kwargs):
    """
    Calculate PSD of a signal
    
    Parameters
    ----------
    x: time-domain signal (V)
    fs: sampling rate (MHz)
    rbw: desired resolution BW (MHz)
    kwargs: for SDE windowing
        - 'wintype'

    Returns
    -------
    p: PSD of x in V^2/Hz
    f: PSD frequencies in MHz

    """
    
    nfft = np.ceil(fs/rbw)

    if not kwargs:
        # kwargs = {
        #     "wintype": "kaiser",
        #     "beta": 25,
        # }
        wintype = 'kaiser'
        kwargs = {"beta": 25}
    
    if wintype == 'kaiser':
        taps = signal.windows.kaiser(nfft, **kwargs)
    elif wintype == 'blackmanharris':
        taps = signal.windows.blackmanharris(nfft, **kwargs)
    elif wintype == 'hann':
        taps = signal.windows.hann(nfft, **kwargs)
    else:
        raise ValueError("psd supports only these windows: 'kaiser', 'blackmanharris', 'hann'")
    
    # detrend must be set to False; otherwise, by default, the mean of the signal is subtracted
    # scaling is set to 'density' by default, but I explicitly call it here. 'density' returns PSD in V^2/Hz if fs is in Hz. Using 'spectrum' seems to lead to inaccurate power estimation.
    f, p = signal.welch(x,
                        fs=fs*1e6,
                        window=taps,
                        detrend=False,
                        return_onesided=False,
                        scaling='density')
    
    f = fft.fftshift(f)/1e6
    p = fft.fftshift(p)
    return (p, f)

def psd_dbm(x: np.ndarray, fs, rbw, low, high, zo=50):
    """
    Calculate power of a signal from its PSD
    
    Parameters
    ----------
    x: signal in Volts
    fs: sampling rate in MHz
    rbw: desired resolution BW in MHz (of the PSD)
    low: low frequency of signal in MHz
    high: high frequency of signal in MHz
    zo: impedance of the system in Ohms

    Returns
    -------
    p_dbm: signal power in dBm
    
    """
    
    [p, f] = psd(x, fs, rbw)
    
    psig = p[(f >= low) & (f <= high)] # integration limits
    fsig = f[(f >= low) & (f <= high)] # for debug only
    p_lin = (psig*rbw*1e6).sum() # convert from V^2/Hz to V^2 (assumes density is constant in a given bin)
    p_dbm = 10*np.log10(p_lin/zo/1e-3)
    
    return p_dbm
    
def rms(x: np.ndarray):
    """
    Calculate rms of a time-domain signal

    Parameters
    ----------
    x: time domain signal (V)

    Returns
    -------
    rms of x (V)

    """

    return np.sqrt(np.vdot(x, x).real/x.size)

def scale_psd(p: np.ndarray, f: np.ndarray, bw, scs, start_sc, num_sc):
    """
    Scale PSD so average signal bin power is 0dBm

    Parameters
    ----------
    p: PSD in V^2/Hz
    f: PSD frequencies in MHz
    bw: signal BW (MHz)
    scs: subcarrier spacing (kHz)
    start_sc: starting subcarrier (index)
    num_sc: number of subcarriers
    
    Returns
    -------
    p: scaled PSD

    """

    nrb = bw*5
    sigl = -nrb*12*scs/1000/2 + start_sc*scs/1000
    sigh = sigl + (num_sc-1)*scs/1000
    psig = p[(f >= sigl) & (f <= sigh)]
    psig = psig.sum()/psig.size
    p = p/psig

    return p