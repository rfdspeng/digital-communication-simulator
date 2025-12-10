# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 10:06 2025

Classes and functions for mixed-signal processing

@author: Ryan Tsai
"""

import numpy as np
from scipy.signal import firls, firwin2, lfilter

class DAC:
    def __init__(self, fs_in: float, fs_eq: float, en_eq: bool=True, osr: int=5, ideal: bool=False):
        self.fs_in = fs_in # DAC sampling rate (MHz)
        self.fs_eq = fs_eq # Equalization bandwidth/signal bandwidth (MHz)
        self.en_eq = en_eq
        self.osr = osr
        self.ideal = ideal # If ideal, implement ideal reconstruction filter using a windowed sinc instead of using ZOH + equalizer

    def fit(self, n_taps_eq: int | None=None):
        """
        Fitted parameters:
            fs_out_: continuous-time "sampling rate" - sampling rate of the DAC output, which is, in reality, a continuous-time waveform
            zoh_: coefficients of the ZOH "filter" - model for a realizable DAC
            g_zoh_: ZOH frequency response
            eq_taps_: equalizer coefficients - for equalizing ZOH response in the signal bandwidth
            fs_eq_: equalization bandwidth - attempts to equalize up to fs_eq_ frequency
            f_, g_: desired equalization response - passed to filter design function (after normalizing frequency)

        """

        self.fs_out_ = self.fs_in*self.osr # CT "sampling rate"
        n_taps_eq = 4096 if n_taps_eq is None else n_taps_eq
        f = np.linspace(0, 1, 4096+1) # normalized frequency (0 to 1)

        # Construct desired filter response
        if not self.ideal: # Equalization filter at fs_in, ZOH at fs_in*osr
            self.zoh_ = np.ones(self.osr)
            # if self.en_eq:
            T = 1/self.fs_in
            # g_zoh = T*np.sinc(f/2)
            g_zoh = np.sinc(f/2)
            g = 1/g_zoh # self.fs_in/2*f*T = f/2
            g[f > self.fs_eq/(self.fs_in/2)] = 0

            self.eq_taps_ = firwin2(n_taps_eq, f, g)
            self.eq_taps_ = self.eq_taps_/self.eq_taps_.sum()
            self.g_zoh_ = g_zoh
            self.f_ = f*self.fs_in/2
            self.g_ = g
            self.fs_filter_ = self.fs_in
                
        else: # "Ideal" reconstruction filter at fs_in*osr
            g = np.ones_like(f)
            g[f > self.fs_eq/(self.fs_in*self.osr/2)] = 0

            self.eq_taps_ = firwin2(n_taps_eq, f, g)
            self.eq_taps_ = self.eq_taps_/self.eq_taps_.sum()*self.osr
            self.f_ = f*self.fs_in*self.osr/2
            self.g_ = g
            self.fs_filter_ = self.fs_out_

        return self

    def transform(self, x: np.ndarray):
        assert x.ndim == 1

        if not self.ideal:
            y_dig = lfilter(self.eq_taps_, 1, x)
            y = np.zeros(y_dig.size*self.osr, dtype=y_dig.dtype)
            y[::self.osr] = y_dig
            y = lfilter(self.zoh_, 1, y)
            t = np.arange(len(y), dtype="float")/self.fs_out_
        else:
            y = np.zeros(x.size*self.osr, dtype=x.dtype)
            y[::self.osr] = x
            y = lfilter(self.eq_taps_, 1, y)
            t = np.arange(len(y), dtype="float")/self.fs_out_

        # t is in us
        return (t, y)