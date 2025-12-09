# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:50:42 2022

Functions for generating a single-carrier waveform

@author: Ryan Tsai
"""

from typing import Literal
import numpy as np
import math
from scipy import signal
from rfdsppy import digital_hw_algo as dighw, calc, rf_estimation as rf_est
import matplotlib.pyplot as plt

# https://pysdr.org/content/pulse_shaping.html

class SCWavGen:

    def __init__(self, symbol_rate: float=3.84, osr: float=8, pulse_shape="rectangular", modulation=("PSK", 2), power: float | None=None, bitwidth: bool | int = False):
        """
        Parameters
        ----------
        modulation: (mod_name, param)
            - ("ASK", M) where M is the number of amplitudes (must be a power of 2)
            - ("PSK", M), where M is the number of phases (must be a power of 2)
            - ("QAM", M), where M is the number of points (16, 64, 256, 1024)
            - ("ASK", 2) and ("PSK", 2) are identical: this is BPSK
        
        """
        self.osr = osr
        self.symbol_rate = symbol_rate
        self.pulse_shape = pulse_shape
        self.mod = modulation[0]
        self.mod_M = modulation[1]
        self.power = power
        self.bitwidth = bitwidth
        self.rng = np.random.default_rng()

        self.b_ = self.pulse_shaping_coefficients()
        self.gd_ = (len(self.b_)-1)/2
        self.fs_ = self.symbol_rate*self.osr
    
    def generate(self, nsym):
        # Generate symbols
        if self.mod == "ASK":
            self.symbols_ = np.round(self.rng.uniform(low=0, high=1, size=nsym)*(self.mod_M-1))*2-(self.mod_M-1) + 1j*np.zeros(nsym)

        elif self.mod == "PSK":
            self.symbols_ = np.exp(1j*2*np.pi*(np.round(self.rng.uniform(low=1, high=self.mod_M, size=nsym))-1)/self.mod_M)
            
            if self.mod_M == 2: # BPSK
                self.symbols_ = self.symbols_.real + 1j*np.zeros(nsym)
            elif self.mod_M == 4: # QPSK
                self.symbols_ = self.symbols_ * np.exp(1j*np.pi/4)

        elif self.mod == "QAM":
            pass

        # Pulse shaping
        pulses = dighw.upsample(self.symbols_, self.osr)
        pulses = signal.lfilter(self.b_, 1, pulses)
        x = pulses

        # Modulate baseband signal
        # if self.mod == "ASK":
        #     x = pulses
        # elif self.mod == "PSK":
        #     pass
        # elif self.mod == "FSK":
        #     # x = np.exp(1j*0.3*np.cumsum(pulses))
        #     x = np.exp(1j*1*np.cumsum(pulses))
        
        if (self.bitwidth == False) and (self.power is not None):
            vrms = calc.dbm2v(self.power, "dBm")
            x = x/calc.rms(x)*vrms

        return x

    def pulse_shaping_coefficients(self):
        if self.pulse_shape == "rectangular":
            b = np.ones(self.osr, dtype="float")/self.osr
        elif self.pulse_shape == "triangular":
            b1 = np.arange(math.ceil(self.osr/2)+1, dtype="float")
            b2 = np.arange(math.ceil(self.osr/2)-1, 0, -1, dtype="float")
            b = np.concatenate((b1, b2))
            b = b/b.sum()
        elif self.pulse_shape == "sawtooth":
            b = np.arange(self.osr, dtype="float")
            b = b/b.sum()
        elif self.pulse_shape == "rc":
            numtaps = round(101*self.osr/8)
            numtaps = numtaps + ((numtaps+1) % 2)
            beta = 0.35
            t = np.arange(-math.floor(numtaps/2), numtaps-math.floor(numtaps/2), dtype="float")
            b = np.sin(np.pi*t/self.osr)/(np.pi*t/self.osr)*np.cos(np.pi*beta*t/self.osr)/(1-(2*beta*t/self.osr)**2)
            b[t == 0] = 1
            b = b/self.osr
        elif self.pulse_shape == "rrc":
            numtaps = round(101*self.osr/8)
            numtaps = numtaps + ((numtaps+1) % 2)
            beta = 0.35
            t = np.arange(-math.floor(numtaps/2), numtaps-math.floor(numtaps/2), dtype="float")
            b = 1/self.osr*(np.sin(np.pi*t/self.osr*(1-beta)) + 4*beta*t/self.osr*np.cos(np.pi*t/self.osr*(1+beta)))/(np.pi*t/self.osr*(1-(4*beta*t/self.osr)**2))
            b[t == 0] = 1/self.osr*(1+beta*(4/np.pi-1))
            b[t == self.osr/4/beta] = beta/self.osr/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/4/beta) + (1-2/np.pi)*np.cos(np.pi/4/beta))
            b[t == -self.osr/4/beta] = beta/self.osr/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/4/beta) + (1-2/np.pi)*np.cos(np.pi/4/beta))
        else:
            b = signal.get_window(self.pulse_shape, self.osr)
            b = b/b.sum()

        return b
    
    def get_demod_params(self) -> dict:
        """
        Return a dictionary containing the waveform parameters needed for demodulation
        
        """
    
        # Output dictionary for demodulation
        cfg = {
            "fs": self.fs_,                                 # Sampling rate (MHz)
            "osr": self.osr,                                # Oversampling ratio
            "symbol_rate": self.symbol_rate,                # Symbol rate (MHz)
            "pulse_shape": self.pulse_shape,                # Pulse shape (string)
            "pulse_shaping_coefficients": self.b_,          # Pulse-shaping coefficients
            "gd": self.gd_,                                 # Pulse-shaping filter group delay - with matched filtering, overall group delay will be twice this
            "modulation": self.mod,                         # Modulation type
            "mod_M": self.mod_M,                            # Number of constellation points
            "symbols": self.symbols_,                       # Original symbols prior to pulse-shaping
        }

        return cfg
    
    def matched_filter_and_sample(self, x: np.ndarray, gd: float | int):
        y_raw = signal.lfilter(self.b_, 1, x)
        y_sampled = y_raw[round(gd+self.gd_)::self.osr]
        return y_sampled

    def calculate_evm(self, y: np.ndarray, en_flat_eq: Literal["gain", "both", False]="gain", \
                      en_fd_eq: bool=False, **kwargs):
        
        x = self.symbols_.copy()
        y = y.copy()
        if en_flat_eq != False:
            g_est = rf_est.est_single_tap_channel(x, y)
            if en_flat_eq == "gain":
                y = y/np.abs(g_est)
            elif en_flat_eq == "both":
                y = y/g_est

        L = min(x.size, y.size)
        e = y[:L]-x[:L]
        
        evm = math.sqrt((np.abs(e)**2).sum()/(np.abs(x[:L])**2).sum())*100

        fig, ax = plt.subplots(dpi=150)
        ax.plot(x.real, x.imag, 'x', markersize=10, label="x")
        ax.plot(y.real, y.imag, 'x', markersize=5, label="y")
        ax.grid()

        return evm