# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 2025

Functions and classes for modeling RF estimation algorithms (FW or HW)

@author: Ryan Tsai
"""

import numpy as np
from scipy import fft, signal
from typing import Literal
import math
import matplotlib.pyplot as plt

def est_tone_amp_phase(x: np.ndarray, fs, f0):
    """
    Estimate amplitude and phase of a sinusoid

    x: sinusoid (real or complex)
    fs = sampling rate (MHz)
    f0 = tone frequency (MHz)
    
    """

    w0 = f0*2*np.pi/fs

    c = (x * np.exp(-1j*w0*np.arange(x.size, dtype="float"))).sum()/x.size

    A = np.abs(c)
    phi = np.angle(c)

    return (A, phi)

def est_single_tap_channel(x: np.ndarray, y: np.ndarray):
    """
    Estimate a memoryless gain/phase response (like a flat fading channel)

    x: reference signal
    y: received signal

    """

    L = min(x.size, y.size)
    # Project y onto x
    return np.vdot(x[:L], y[:L])/np.vdot(x[:L], x[:L])

def comp_single_tap_channel(x: np.ndarray, y: np.ndarray):
    g = est_single_tap_channel(x, y)
    return y.copy()/g

def est_delay(x: np.ndarray, y: np.ndarray):
    """
    Calculate delay of y w.r.t. x
    Positive delay means y is delayed relative to x (this is the typical case)
    Negative delay means y is advanced
    
    """
    corr = np.abs(signal.correlate(y, x))
    lags = signal.correlation_lags(y.size, x.size)

    return lags[corr.argmax()].item()

def time_align(x: np.ndarray, y: np.ndarray):
    delay = est_delay(x, y)

    return (x.copy(), y.copy()[delay:])

def sde(x: np.ndarray, fs, rbw, window: str | tuple | float | None=None, **kwargs):
    """
    Estimate the spectral density of a signal
    
    Parameters
    ----------
    x: time-domain signal (V)
    fs: sampling rate (MHz)
    rbw: desired resolution BW (MHz)
    window: FFT windowing

    Returns
    -------
    p: PSD of x in V^2/Hz
    f: PSD frequencies in MHz

    """
    
    # Required FFT size for the desired resolution
    nfft = np.ceil(fs/rbw)

    if window is None:
        window = 10.0 # Kaiser window beta parameter
    
    win = signal.get_window(window, nfft)
    
    # detrend must be set to False; otherwise, by default, the mean of the signal is subtracted
    # scaling is set to 'density' by default, but I explicitly call it here. 'density' returns PSD in V^2/Hz if fs is in Hz.
    f, p = signal.welch(x,
                        fs=fs*1e6,
                        window=win,
                        detrend=False,
                        return_onesided=False,
                        scaling='density')
    
    f = fft.fftshift(f)/1e6
    p = fft.fftshift(p)

    if kwargs.get("en_plot", False):
        fig, ax = plt.subplots(dpi=100)
        ax.plot(f, 10*np.log10(p))
        ax.set_title(kwargs.get("title", "PSD"))
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("PSD (dBm/Hz)")
        ax.grid()

    return (p, f)