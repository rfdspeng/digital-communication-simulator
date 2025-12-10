# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 2025

Functions and classes for modeling beamforming

@author: Ryan Tsai
"""

import numpy as np
from scipy import fft, signal
from typing import Literal
import math
import matplotlib.pyplot as plt

class ULA:

    def __init__(self, Nr: float | int, d: float=0.5):
        """
        ULA: uniform linear array

        Parameters
        ----------
        Nr: number of elements
        d: distance between elements as a fraction of wavelength

        """

        self.Nr = Nr
        self.d = d
    
    def fit(self, theta: float) -> None:
        """
        Parameters
        ----------
        theta: direction of arrival in radians. 0 degrees = along the line of the linear array, with theta increasing clockwise. Reference element is left-most.

        """

        self.theta_ = theta
        self.s_ = np.exp(1j*2*np.pi*self.d*np.sin(theta)*np.arange(self.Nr, dtype="float")).reshape((self.Nr, 1)) # Steering vector

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().reshape((1, x.size)) # Reshape x to row vector
        y = self.s_ @ x

        return y