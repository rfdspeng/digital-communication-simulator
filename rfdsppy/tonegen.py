# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 08:55:31 2023

Functions to generate tones

@author: Ryan Tsai
"""

import numpy as np
from fractions import Fraction
from rfdsppy import calc
from typing import Literal

def tonegen(fs: float, fc: float, cossin: Literal["cos", "sin", "exp"]="cos", theta0: float=0, nsamp: int | float | None=None, power: int | float | None=None):
    """
    Generates real or complex tone at fc with starting phase of theta0
    x = cos(wn + theta0) or sin(wn + theta0) or exp(j(wn + theta0))

    Parameters
    ----------
    nsamp : number of samples
    fs : sampling rate
    fc : tone frequency
    cossin : 'cos', 'sin', or 'exp'
    theta0 : starting phase in radians
    power : power in a 50Ohm system (dBm)

    Returns
    -------
    x : real or complex tone

    """
    
    # Convert theta0 from degrees to radians
    # theta0 = theta0*np.pi/180

    wc = 2*np.pi*fc/fs # convert from CT frequency to DT frequency (rad/sample)

    # Auto-calculate nsamp if not provided - calculate nsamp to approximate one hundred periods of the sinusoid
    if nsamp is None:
        nsamp = Fraction(fc/fs).limit_denominator(2**16).denominator * 100
    
    n = np.arange(nsamp)
    if cossin == 'cos':
        x = np.cos(wc*n + theta0)
    elif cossin == 'sin':
        x = np.sin(wc*n + theta0)
    elif cossin == 'exp':
        x = np.exp(1j*(wc*n + theta0))
    else:
        raise Exception("'cossin' valid options are 'cos', 'sin', 'exp'")

    if power is not None:
        vrms = calc.dbm2v(power, unit="dBm")
        x = x/calc.rms(x)*vrms

    return x