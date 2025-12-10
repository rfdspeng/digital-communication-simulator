# -*- coding: utf-8 -*-
"""
Created on Tues Aug 26 10:55:47 2025

Classes and functions for working with CT signals and systems

@author: Ryan Tsai
"""

import numpy as np

class CTPeriodicSigGen:
    """
    Generate a continuous-time periodic signal

    """

    def __init__(self, sigtype: str = "pulse", **kwargs):
        assert sigtype in ["pulse", "triangle", "half-rect-sine", "full-rect-sine"]
        self.sigtype = sigtype
        self.kwargs = kwargs
    
    def calc_fourier_series_coef(self):
        if self.sigtype == "pulse":
            self.A = self.kwargs.get("A", 1)
            self.tau = self.kwargs.get("tau", 0.5)
            self.To = self.kwargs.get("To", 1)
            self.fo = self.kwargs.get("fo", 1/self.To)
            self.max_harmonic = self.kwargs.get("max_harmonic", 100)
            
            self.sidelobe_ = self.To/self.tau
            self.k_ = np.arange(-self.max_harmonic, self.max_harmonic+1).astype("float")

            if self.tau/self.To == 0.5:
                self.Xk_ = self.A/1j/np.pi/self.k_
                self.Xk_[np.mod(self.k_, 2) == 0] = 0
                self.Xk_[self.k_ == 0] = self.A/2
            else:
                self.Xk_ = self.A*self.tau/self.To * np.sinc(self.k_*self.fo*self.tau) \
                    * np.exp(-1j*np.pi*self.k_*self.fo*self.tau)

        elif self.sigtype == "triangle":
            self.A = self.kwargs.get("A", 1)
            self.To = self.kwargs.get("To", 1)
            self.fo = self.kwargs.get("fo", 1/self.To)
            self.max_harmonic = self.kwargs.get("max_harmonic", 100)

            self.k_ = np.arange(-self.max_harmonic, self.max_harmonic+1).astype("float")

            self.Xk_ = np.zeros_like(self.k_)
            self.Xk_ = -2*self.A/np.pi**2/self.k_**2
            self.Xk_[np.mod(self.k_, 2) == 0] = 0
            self.Xk_[self.k_ == 0] = self.A/2
        
        self.f_ = self.k_*self.fo

        return (self.f_.copy(), self.Xk_.copy())
    
    def gen_sig(self, t: np.ndarray | None=None, max_harmonic=None):
        self.calc_fourier_series_coef()
        
        t = t.copy() if t is not None else np.linspace(0, 2*self.To, 1000)

        if max_harmonic is None:
            max_harmonic = self.k_.max()

        i = (self.k_ >= -max_harmonic) & (self.k_ <= max_harmonic)
        kr = self.k_[i].copy()
        Xr = self.Xk_[i].copy()
        kr_mat = np.tile(kr[np.newaxis, :], [len(t), 1])
        t_mat = np.tile(t[:, np.newaxis], [1, len(kr)])
        cmplx_sins = np.exp(1j*2*np.pi*kr_mat*self.fo*t_mat)
        Xr = Xr[:, np.newaxis]
        xr = (cmplx_sins @ Xr).real.squeeze()

        if self.sigtype == "pulse":
            x = self.A*(np.mod(t, self.To) <= self.tau)
        elif self.sigtype == "triangle":
            x = np.zeros_like(t)
            # x[t < self.To/2] = 2*t[t < self.To/2]/self.To
            # x[t >= self.To/2] = 2*(self.To-t[t >= self.To/2])/self.To
            x[np.mod(t, self.To) < self.To/2] = 2*np.mod(t, self.To)[np.mod(t, self.To) < self.To/2]/self.To
            x[np.mod(t, self.To) >= self.To/2] = 2*(self.To-np.mod(t, self.To)[np.mod(t, self.To) >= self.To/2])/self.To

        return (t, x, xr)