# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 2025

Functions and classes for modeling beamforming

@author: Ryan Tsai
"""

import numpy as np
from scipy import fft, signal
from typing import Literal, Iterable
import math
import matplotlib.pyplot as plt
from digcommpy import rf_analog

class ULA:

    def __init__(self, Nr: float | int, theta: float, d: float=0.5):
        """
        ULA: For simulating the signals received at each antenna element in a uniform linear array.
        No beamforming is applied; the output of transform is a 2D matrix representing the signals
        for each element.

        Parameters
        ----------
        Nr: number of elements
        theta: direction of arrival in radians
        d: distance between elements as a fraction of wavelength

        """

        self.Nr = Nr
        self.d = d
        self.theta = theta
    
    def fit(self) -> None:
        self.s_ = np.exp(1j*2*np.pi*self.d*np.sin(self.theta)*np.arange(self.Nr, dtype="float")).reshape((self.Nr, 1)) # Steering vector

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().reshape((1, x.size)) # Reshape x to row vector
        y = self.s_ @ x

        return y
    
class ULABeamform:

    def __init__(self, Nr: float | int, theta: float, d: float=0.5):
        """
        ULABeamform: For beamforming the signals received at the antenna elements.
        The input to transform is a 2D matrix representing the signals, and the output
        of transform is a 1D vector representing the beamformed signal.

        Parameters
        ----------
        Nr: number of elements
        theta: direction of arrival in radians
        d: distance between elements as a fraction of wavelength

        """

        self.Nr = Nr
        self.d = d
        self.theta = theta
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
    
    def fit(self) -> None:
        self.w_ = np.exp(1j*2*np.pi*self.d*np.sin(self.theta)*np.arange(self.Nr, dtype="float")).reshape((self.Nr, 1)) # Weights

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2, "Input to transform must be 2D"
        assert x.shape[0] == self.Nr, "Row dimension of input must equal the number of antenna elements"
        x = x.copy()
        y = (self.w_.conjugate().T @ x).squeeze()

        return y
    
# def ula_doa(x: np.ndarray, Nr: float | int, theta: float, d: float=0.5, theta_sweep: Iterable | np.ndarray | None=None):
#     # Add option for AWGN - fs, bw, power
#     # Add option to autocalculate AWGN

#     if theta_sweep is None:
#         # theta_sweep = np.arange(-np.pi, np.pi, 1*np.pi/180)
#         theta_sweep = np.linspace(-1*np.pi, np.pi, 1000)
    
#     ula = ULA(Nr, theta, d)
#     ula.fit()
#     X = ula.transform(x.copy())

#     ula_bf = ULABeamform(Nr, 0, d)

#     powers = np.zeros(len(theta_sweep), dtype="float")
#     for tdx, theta_s in enumerate(theta_sweep):
#         ula_bf.set_params(theta=theta_s)
#         ula_bf.fit()
#         y = ula_bf.transform(X)

#         # powers[tdx] = np.vdot(y, y).real.item()
#         powers[tdx] = np.var(y)
    
#     return (theta_sweep, powers)

def ula_doa(x: np.ndarray, ula: ULA, ula_bf: ULABeamform, awgn: rf_analog.AWGN, theta_sweep: Iterable | np.ndarray | None=None):
    if theta_sweep is None:
        # theta_sweep = np.arange(-np.pi, np.pi, 1*np.pi/180)
        theta_sweep = np.linspace(-1*np.pi, np.pi, 1000)
    
    ula.fit()
    X = ula.transform(x.copy())
    X = awgn.transform(X)

    powers = np.zeros(len(theta_sweep), dtype="float")
    for tdx, theta_s in enumerate(theta_sweep):
        ula_bf.set_params(theta=theta_s)
        ula_bf.fit()
        y = ula_bf.transform(X)

        # powers[tdx] = np.vdot(y, y).real.item()
        powers[tdx] = 10*np.log10(np.var(y))
    
    powers -= powers.max()
    
    return (theta_sweep, powers)