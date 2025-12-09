# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:13:34 2023

Functions to design digital filters

@author: Ryan Tsai
"""

import math
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft
from typing import Literal

def firls_rate_change(updn: Literal["up", "down"], ntaps, obw, fs_in, R, passband_ripple_spec_db=0.1, stopband_rej_spec_db=50, en_plot=False):
    """
    Description
    -----------
    Generates AAF/interpolation filter coefficients

    Parameters
    ----------
    updn : 'up' or 'down' to specify upsampling or downsampling
    ntaps : filter length
    obw : occupied BW of desired lowpass signal
    fs_in : input sampling rate
    R : integer rate change
    passband_ripple_spec_db : passband ripple spec in dB
    stopband_rej_spec_db : stopband rejection spec in dB

    Returns
    -------
    b : AAF/interpolation filter coefficients

    """
    
    # Determine filter sampling rate
    if updn == 'up':
        fs = fs_in*R 
    elif updn == 'down':
        fs = fs_in
    
    # Determine passband and stopband
    passband = (obw/2)/(fs/2)
    stopband = (fs/R-obw/2)/(fs/2)
    bands = [0, passband, stopband, 1]
    amps = [1, 1, 0, 0]
    
    # Determine firls weighting
    passband_lin_dev = 1-10**(-passband_ripple_spec_db/20)
    stopband_lin_dev = 10**(-stopband_rej_spec_db/20)
    weights = np.array([passband_lin_dev, stopband_lin_dev])
    # weights = max(weights)/weights
    # weights = weights**2
    weights = (weights.max()/weights)**2
    
    # Generate filter coefficients
    b = signal.firls(ntaps, bands, amps, weight=weights, fs=2)
    w, h = signal.freqz(b, fs=2)
    # h_pb = 20*np.log10(abs(h[w <= passband]))
    # h_sb = -20*np.log10(abs(h[w >= stopband]))
    # wc_pb_ripple = max(abs(h_pb))
    # wc_sb_rej = min(h_sb)
    h_pb = 20*np.log10(np.abs(h[w <= passband]))
    h_sb = -20*np.log10(np.abs(h[w >= stopband]))
    wc_pb_ripple = np.abs(h_pb).max() # dB
    wc_sb_rej = h_sb.min() # dB
    
    print('digital_filter_design.firls_rate_change()')
    print('Largest passband ripple (dB) = ' + str(round(wc_pb_ripple,3)))
    print('Smallest stopband rejection (dB) = ' + str(round(wc_sb_rej,1)))
    print('\n\n')
    
    if en_plot:
        fig, axs = plt.subplots(nrows=2, dpi=100, figsize=(8, 6))
        axs[0].plot(w, 20*np.log10(np.abs(h)))
        axs[1].plot(w, np.angle(h))
        axs[0].set_title("Magnitude Response (dB)")
        axs[1].set_title("Phase Response (rad)")
        axs[0].grid()
        axs[1].grid()

    return b

def iir_bbf(wp, ws, gpass, gstop, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates IIR filter coefficients based on desired passband/stopband magnitude response

    Parameters
    ----------
    wp: passband cutoff. This must be normalized to fs (which is 2 by default).
    ws: stopband cutoff. This must be normalized to fs (which is 2 by default).
    gpass: maximum passband loss (dB)
    gstop: minimum stopband attenuation (dB)
    kwargs
        ftype: "butter", etc.
        fs: sampling rate (by default, 2)
    
    """
    
    ftype = kwargs.get("ftype", "butter")
    fs = kwargs.get("fs", 2)
    b, a = signal.iirdesign(wp, ws, gpass, gstop, ftype=ftype, fs=fs)    

    w, h = signal.freqz(b, a, fs=fs)

    h_pb = 20*np.log10(np.abs(h[w <= wp]))
    h_sb = -20*np.log10(np.abs(h[w >= ws]))
    wc_pb_ripple = np.abs(h_pb).max() # dB
    wc_sb_rej = h_sb.min() # dB

    print("digital_filter_design.iir_bbf()")
    print(f"Filter order = {a.size}")
    print('Maximum passband loss (dB) = ' + str(round(wc_pb_ripple,3)))
    print('Minimum stopband attenuation (dB) = ' + str(round(wc_sb_rej,1)))
    print('\n\n')
    
    if kwargs.get("en_plot", False):
        fig, axs = plt.subplots(nrows=2, dpi=100, figsize=(6, 8))
        axs[0].plot(w, 20*np.log10(np.abs(h)))
        axs[1].plot(w, np.angle(h))
        axs[0].set_title("Magnitude Response (dB)")
        axs[1].set_title("Phase Response (rad)")
        axs[0].grid()
        axs[1].grid()

    return (b, a)

def fir_equalizer(H: np.ndarray, omega: np.ndarray, L, W: np.ndarray | None=None, en_plot=False):
    """
    H: desired complex frequency response
    omega: DT frequencies
    L: filter length
    W: weights
    
    """
    
    H = H.reshape((H.size, 1))
    if W is not None:
        W = W.reshape((W.size, 1))
        H = H*W

    Omega = [np.exp(1j*omega_i*np.arange(L, dtype="float")).reshape((L, 1)) for omega_i in omega]
    Omega = np.array(Omega).transpose()

    h = linalg.pinv(Omega.transpose().conjugate @ Omega) @ Omega.transpose().conjugate @ H

    h = h.squeeze()

    if en_plot:
        w, H_fitted = signal.freqz(h, worN=omega)

        fig, axs = plt.subplots(nrows=2, dpi=100, figsize=(6, 8))
        axs[0].plot(omega, 20*np.log10(np.abs(H)), label="Desired")
        axs[0].plot(w, 20*np.log10(np.abs(H_fitted)), label="Fitted")
        axs[1].plot(omega, np.angle(H)*180/np.pi, label="Desired")
        axs[1].plot(w, np.angle(H_fitted)*180/np.pi, label="Fitted")
        axs[0].set_title("Magnitude Response (dB)")
        axs[1].set_title("Phase Response (Degrees)")
        axs[0].grid()
        axs[1].grid()
        # axs[0].set_xlim(left=0, right=wp*3)
        # axs[1].set_xlim(left=0, right=wp*3)
        # # axs[0].set_ylim(bottom=-10, top=0)
        # axs[0].vlines([wp], ymin=ep.min(), ymax=ep.max(), colors='r')
        # axs[1].vlines([wp], ymin=theta_deg.min(), ymax=theta_deg.max(), colors='r')

    return h