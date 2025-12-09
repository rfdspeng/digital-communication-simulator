# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 2025

Functions and classes for modeling RF analog blocks and impairments

@author: Ryan Tsai
"""

import numpy as np
import matplotlib.pyplot as plt
from rfdsppy import calc, digital_hw_algo as dighw
import scipy.fft
import math
from typing import Literal
from scipy import signal

class AWGN:
    """
    class AWGN

    Can generate either real or complex noise
    
    """

    def __init__(self, power, bw, fs):
        """
        power: power in a 50Ohm system (dBm). This is power within the signal bandwidth.
        bw: signal bandwidth (MHz)
        fs: sampling rate (MHz)
        
        If the input signal to transform() is real, then the noise generated is real.
        If the input signal to transform() is complex, then the noise generated is complex, and the power is divided between I and Q.
        
        """

        self.rng = np.random.default_rng()
        self.vrms = calc.dbm2v(power, unit="dBm")*np.sqrt(fs/bw)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(x):
            n = self.rng.normal(loc=0, scale=self.vrms/np.sqrt(2), size=x.size) + \
                1j*self.rng.normal(loc=0, scale=self.vrms/np.sqrt(2), size=x.size)
        else:
            n = self.rng.normal(loc=0, scale=self.vrms, size=x.size)

        return x + n

class FlickerNoise:
    ""

class PhaseNoise:
    """
    Lowpass equivalent model
    
    """

    def __init__(self, l0, lfloor, bpll, fcorner, fs):
        """
        l0 = inband noise floor (dBc/Hz)
        lfloor = out-of-band noise floor (dBc/Hz)
        bpll = PLL 3dB BW (MHz)
        fcorner = flicker noise corner frequency (MHz)
        fs = sampling rate (MHz)
        
        """
        self.l0 = l0
        # self.l0_lin = 10**(l0/10)
        self.lfloor = lfloor
        # self.lfloor_lin = 10**(lfloor/10)
        self.bpll = bpll
        self.fcorner = fcorner
        self.fs = fs
        self.rng = np.random.default_rng()
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        # One-sided PSD
        # f = np.linspace(0, self.fs/2, x.size+2 + (x.size+1)%2)[1:]
        f = np.linspace(0, self.fs/2, math.ceil(x.size/2)+2 + (math.ceil(x.size/2)+1)%2)[1:]
        l0_lin = 10**(self.l0/10)
        lfloor_lin = 10**(self.lfloor/10)
        fbin = (f[1]-f[0])*1e6 # Hz
        # l0_lin = 10**(self.l0/10)*fbin
        # lfloor_lin = 10**(self.lfloor/10)*fbin
        P = self.bpll**2*l0_lin/(self.bpll**2 + f**2)*(1+self.fcorner/f) + lfloor_lin
        phi = self.rng.uniform(low=-np.pi, high=np.pi, size=P.size)

        # FT
        F = np.concatenate(([0], np.sqrt(fbin*P[:-1]/2), np.sqrt(fbin*P[::-1]/2)))
        phi2 = np.concatenate(([0], np.exp(1j*phi[:-1]), np.exp(-1j*phi[::-1])))
        F = F*phi2*F.size

        # IFFT
        theta = scipy.fft.ifft(F)
        theta = theta.real
        # theta = theta*4343*np.sqrt(1000)

        self.theta_ = theta
        self.P_ = P # linear, relative to carrier
        self.Pf_ = f # MHz
        self.F_ = F
        self.Ff_ = np.linspace(0, self.fs, F.size)
        self.fbin_ = fbin # Hz

        return x*np.exp(1j*theta[:x.size])
    
    def plot(self):
        fig, ax = plt.subplots(dpi=100)

        ax.semilogx(self.Pf_, 10*np.log10(self.P_))
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Phase Noise (dBc/Hz)")
        ax.set_title(f"L0={self.l0} dBc/Hz, Lfloor={self.lfloor} dBc/Hz, Bpll={self.bpll} MHz, fcorner={self.fcorner} MHz")
        ax.grid()
        return (fig, ax)


    def calculate_ipn(self, fmin=None, fmax=None):
        if fmin is None:
            fmin = self.Pf_.min()
        if fmax is None:
            fmax = self.Pf_.max()

        return 10*np.log10((self.P_[np.logical_and(self.Pf_ >= fmin, self.Pf_ <= fmax)]*self.fbin_).sum())

class DAC:
    def __init__(self, R: float | int):
        """
        R = upsampling rate; integer
        
        """

        self.R = R
        self.b_ = np.ones(R)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x = dighw.upsample(x, self.R)
        x = signal.lfilter(self.b_, 1, x)

        return x
    
class BBF:
    def __init__(self, b: np.ndarray, a: np.ndarray):
        if (b.ndim == 1) and (a.ndim == 1):
            self.b = b.copy()
            self.a = a.copy()
            self.sep_i_q_ = False
        elif (b.ndim == 2) and (a.ndim == 2):
            # Assume b is 2 x M
            # Assume a is 2 x N
            # First row is I BBF, second row is Q BBF
            assert b.shape[0] == 2
            assert a.shape[0] == 2
            self.b_i = b[0, :].copy()
            self.a_i = a[0, :].copy()
            self.b_q = b[1, :].copy()
            self.a_q = a[1, :].copy()
            self.sep_i_q_ = True
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.sep_i_q_:
            I = signal.lfilter(self.b_i, self.a_i, x.real)
            Q = signal.lfilter(self.b_q, self.a_q, x.imag)
            return I + 1j*Q
        else:
            return signal.lfilter(self.b, self.a, x)

class IQDownconverter:
    def __init__(self, theta=0, ep=0, mode: Literal["balanced", "one-sided"]="balanced"):
        """
        theta = IQ phase mismatch in radians
        ep = IQ gain mismatch (linear)

        """
        self.theta = theta
        self.ep = ep
        self.mode = mode
    
        if self.mode == "balanced":
            self.coeff_sig_ = np.cos(theta/2) - 1j*ep/2*np.sin(theta/2)
            self.coeff_img_ = ep/2*np.cos(theta/2) + 1j*np.sin(theta/2)
        elif self.mode == "one-sided":
            pass
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x*self.coeff_sig_ + x.conjugate()*self.coeff_img_

class IQUpconverter:
    """
    class IQUpconverter

    Lowpass equivalent model of quadrature upconverter
    
    """
    def __init__(self, theta=0, ep=0, mode: Literal["balanced", "one-sided"]="balanced"):
        """
        theta = IQ phase mismatch in radians
        ep = IQ gain mismatch (linear)

        """
        self.theta = theta
        self.ep = ep
        self.mode = mode
    
        if self.mode == "balanced":
            self.LO_I_ = (1 + self.ep/2)*np.exp(1j*self.theta/2)
            self.LO_Q_ = -1j*(1 - self.ep/2)*np.exp(-1j*self.theta/2)
        elif self.mode == "one-sided":
            self.LO_I_ = (1 + self.ep)*np.exp(1j*self.theta)
            self.LO_Q_ = -1j
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x.real*self.LO_I_ - x.imag*self.LO_Q_
    
def rapp_saleh_model(x: np.ndarray, cfg: dict | None = None):
    """
    Rapp-Saleh model (baseband equivalent of RF model)

    https://www.mathworks.com/help/comm/ref/memorylessnonlinearity.html
    - Rapp model for AMAM
    - Saleh model for AMPM

    Parameters
    ----------
    x: complex baseband time-domain signal (V)
    cfg: dictionary
        - 'g': gain in the linear region (Rapp)
        - 'smoothness': smoothness factor (Rapp)
        - 'osat': output saturation level (Rapp)
        - 'a': AMPM alpha (Saleh)
        - 'b': AMPM beta (Saleh)
        - 'en_plot'
    
    Returns
    -------
    y: the output of the Rapp-Saleh model

    """
    
    cfg = cfg if cfg else {}

    g = cfg.get("g", 10**(30/20))
    s = 2*cfg.get("smoothness", 2)
    osat = cfg.get("osat", 25)
    a = cfg.get("a", 0.5)
    b = cfg.get("b", 10)
    
    # Apply model to x
    env = np.abs(x)
    ph = np.angle(x)
    env_y = g*env/(1+(g*env/osat)**s)**(1/s)
    ampm = (a*env**2)/(1+b*env**2)
    y = env_y*np.exp(1j*ph)*np.exp(1j*ampm)
    
    if cfg.get('en_plot', False):
        plt.figure()
        plt.plot(env, env_y, '.')
        plt.title('Rapp-Saleh AMAM', {'fontsize':40})
        plt.xlabel("X (V)", {'fontsize':30})
        plt.ylabel("Y (V)", {'fontsize':30})
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        
        plt.figure()
        plt.plot(env, ampm*180/np.pi, '.')
        plt.title('Rapp-Saleh AMPM', {'fontsize':40})
        plt.xlabel("X (V)", {'fontsize':30})
        plt.ylabel("AMPM (Degrees)", {'fontsize':30})
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
    
    return y

class RappSaleh:
    """
    Rapp-Saleh model (lowpass equivalent amplifier model)

    https://www.mathworks.com/help/comm/ref/memorylessnonlinearity.html
    - Rapp model for AMAM
    - Saleh model for AMPM
    
    Constructor parameters
    ----------------------
    cfg: dictionary
        - 'g': gain in the linear region (Rapp)
        - 'smoothness': smoothness factor (Rapp)
        - 'osat': output saturation level (Rapp)
        - 'a': AMPM alpha (Saleh)
        - 'b': AMPM beta (Saleh)

    Methods
    -------
    transform(x)

    """
    
    def __init__(self, cfg: dict | None = None):
        cfg = cfg if cfg else {}

        # If cfg isn't provided, assume peak input is 1
        self.g = cfg.get("g", 1)
        self.s = 2*cfg.get("smoothness", 2)
        self.osat = cfg.get("osat", 8.5)
        self.a = cfg.get("a", 1)
        self.b = cfg.get("b", 30)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply model to input waveform

        Parameters
        ----------
        x: complex baseband time-domain signal (V)

        Returns
        -------
        y: transformed signal (V)

        """
        env = np.abs(x)
        ph = np.angle(x)

        env_y = self.g*env/(1+(self.g*env/self.osat)**self.s)**(1/self.s)
        ampm = (self.a*env**2)/(1+self.b*env**2)
        y = env_y*np.exp(1j*ph)*np.exp(1j*ampm)

        return y

class IP2_IP3_Nonlinearity:
    """
    Receiver model
    
    """

    def __init__(self, IIP2, IIP3):
        """
        IIP2: in dBm (power of one tone)
        IIP3: in dBm (power of one tone)
        
        
        """
        

        self.IIP2 = IIP2
        self.IIP3 = IIP3