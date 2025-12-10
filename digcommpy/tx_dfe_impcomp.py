# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 2025

Functions and classes for modeling RF Tx digital front end

@author: Ryan Tsai
"""

import numpy as np
from typing import Literal

class TxIQMC:
    """
    Tx frequency-independent IQ mismatch compensation

    """
    
    def __init__(self, ep, theta, mode: Literal["balanced", "one-sided"]="balanced"):
        """
        ep = gain mismatch (linear)
        theta = phase mismatch (radians)

        IQ mismatch model:
        LO_I = (1 + ep/2)*cos(wc*t + theta/2)
        LO_Q = (1 - ep/2)*sin(wc*t - theta/2)
        """

        self.ep = ep
        self.theta = theta
        self.mode = mode

        if self.mode == "balanced":
            self.gii_ = 1/((1+self.ep/2)*np.cos(self.theta/2)*(1-np.tan(self.theta/2)**2))
            self.gqi_ = self.gii_*-np.tan(self.theta/2)
            self.gqq_ = 1/((1-self.ep/2)*np.cos(self.theta/2)*(1-np.tan(self.theta/2)**2))
            self.giq_ = self.gqq_*-np.tan(self.theta/2)
        elif self.mode == "one-sided":
            self.gii_ = 1/((1+self.ep)*np.cos(self.theta))
            self.giq_ = -np.tan(self.theta)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "balanced":
            I = self.gii_*x.real + self.gqi_*x.imag
            Q = self.gqq_*x.imag + self.giq_*x.real
        elif self.mode == "one-sided":
            I = self.gii_*x.real
            Q = x.imag +self.giq_*x.real

        return I + 1j*Q