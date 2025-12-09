# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 08:56:27 2023

Functions to implement an IQ transceiver model

@author: Ryan Tsai
"""

import math
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fft
import tonegen

def upconvert(x,fs,fc,common_phase=0,lo_phase_mismatch=0,lo_gain_mismatch=0):
    """
    Upconverts complex x to fc
    
    x_bb = r*exp(1j*phi)
    x_rf = r*cos(wc*t + phi)

    Parameters
    ----------
    x : complex input signal
    fs : sampling rate of x
    fc : desired upconversion frequency
    common_phase : common phase for LO paths in degrees
    lo_phase_mismatch : phase mismatch for LO paths in degrees
    lo_gain_mismatch : linear gain mismatch for LO paths

    Returns
    -------
    y : real upconverted output signal

    """
    
    # Separate complex x into I and Q
    I = np.real(x)
    Q = np.imag(x)
    
    # Generate LOs
    lo_i, lo_q = generate_lo(len(x),fs,fc,common_phase=common_phase,lo_phase_mismatch=lo_phase_mismatch,lo_gain_mismatch=lo_gain_mismatch)
    
    # Upconvert
    y = I*lo_i + Q*lo_q

    return y

def downconvert(x,fs,fc,common_phase=0,lo_phase_mismatch=0,lo_gain_mismatch=0):
    """
    Downconverts real x

    Parameters
    ----------
    x : real input signal
    fs : sampling rate
    fc : downvert from fc frequency
    common_phase : common phase for LO paths in degrees
    lo_phase_mismatch : phase mismatch for LO paths in degrees
    lo_gain_mismatch : linear gain mismatch for LO paths

    Returns
    -------
    y : complex output signal

    """
    
    # Generate LOs
    lo_i, lo_q = generate_lo(len(x),fs,fc,common_phase=common_phase,lo_phase_mismatch=lo_phase_mismatch,lo_gain_mismatch=lo_gain_mismatch)
    #lo_i = 2*lo_i
    #lo_q = 2*lo_q
    
    # Downconvert
    I = x*lo_i
    Q = x*lo_q
    y = I + 1j*Q
    
    return y

def generate_lo(nsamp,fs,fc,common_phase=0,lo_phase_mismatch=0,lo_gain_mismatch=0):
    """
    Generates IQ LOs
    LO I = (1 + ep/2) * cos(wn + common phase + phase mismatch/2)
    LO Q = (1 - ep/2) * cos(wn + common phase - phase mismatch/2)
    
    Parameters
    ----------
    nsamp : number of samples
    fs : sampling rate
    fc : LO frequency
    common_phase : common phase for LO paths in degrees
    lo_phase_mismatch : phase mismatch for LO paths in degrees
    lo_gain_mismatch : linear gain mismatch for LO paths

    Returns
    -------
    lo_i : LO I (numpy array)
    lo_q : LO Q (numpy array)

    """
    
    # Convert phases from degrees to radians
    theta_com = common_phase*np.pi/180
    theta = lo_phase_mismatch*np.pi/180
    ep = lo_gain_mismatch
    
    wc = np.pi*fc/(fs/2) # digital LO frequency in rad/s
    n = np.array(range(nsamp))
    lo_i = (1+ep/2)*np.cos(wc*n + theta_com + theta/2)
    lo_q = -(1-ep/2)*np.sin(wc*n + theta_com - theta/2)
    
    """
    theta_i = common_phase+lo_phase_mismatch/2
    theta_q = common_phase-lo_phase_mismatch/2
    lo_i = tonegen.generate_real_tone(nsamp,fs,fc,cossin='cos',theta0=theta_i)
    lo_q = tonegen.generate_real_tone(nsamp,fs,fc,cossin='sin',theta0=theta_q)
    lo_i = 2*(1+ep/2)*lo_i
    lo_q = -2*(1-ep/2)*lo_q
    """
    
    return (lo_i,lo_q)