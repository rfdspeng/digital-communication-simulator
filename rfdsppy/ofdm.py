# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:50:42 2022

Functions for OFDM waveforms

@author: Ryan Tsai
"""

import numpy as np
from scipy import signal
from scipy import fft
import matplotlib.pyplot as plt
from rfdsppy import calc
from rfdsppy.digital_modulation import modulation_mapper
from rfdsppy import rf_estimation as rf_est
from typing import Literal

def ofdm_wavgen(nsym: int=14, bw: int=10, scs: int=15, 
                num_sc: int | None=None, 
                start_sc=0, modorder=4, 
                en_tprecode: bool=False, 
                osr: int=1, 
                ncp: float=0.07, 
                wola: float=0.0, 
                seed: int | None=0):
    """
    OFDM waveform generator

    Parameters
    ----------
    nsym: number of symbols to generate
    bw: signal bandwidth (MHz)
    scs: subcarrier spacing (kHz)
    num_sc: number of occupied subcarriers
    start_sc: starting subcarrier index
    modorder: modulation order (4, 16, 64, 256)
    en_tprecode: enable/disable transform precoding to switch b/w DFT-s-OFDM and CP-OFDM
    osr: oversampling ratio
    ncp: cyclic prefix length as a fraction of FFT size
    wola: WOLA length as a fraction of FFT size
    seed: RNG seed for repeatability

    Returns
    -------
    x: waveform with WOLA applied
    x_standard: waveform without WOLA applied
    cfg: dictionary that contains the waveform information for demodulation

    """

    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    nrb = round(bw*5*15/scs) # max number of resource blocks
    nfft = (2**np.ceil(np.log2(nrb*12))).astype(np.uint).item() # matural FFT size (osr=1)
    nfft = nfft*osr # oversampling
    ncp = round(nfft*ncp/100) # number of samples for cyclic prefix
    sym_len = nfft+ncp # number of samples per symbol
    fs = nfft*scs/1000 # waveform sampling rate (MHz)
    bps = round(np.log2(modorder)) # bits per symbol
    wola_len = round(nfft*wola/100) # number of samples for WOLA window
    wola_len = wola_len + (wola_len % 2) # needs to be even (half outside of symbol, half inside of symbol)
    b = OFDMWavGen.generate_wola_window(wola_len, sym_len) # WOLA window (number of samples = wola_len + sum_len)

    # Automatically calculate the number of subcarriers to allocate if none is provided
    if num_sc is None:
        num_sc = nrb*12 - start_sc
    
    x_standard = np.zeros(nsym*sym_len) + 1j*np.zeros(nsym*sym_len) # no WOLA
    x = np.zeros(nsym*sym_len+wola_len) + 1j*np.zeros(nsym*sym_len+wola_len) # with WOLA
    for sdx in range(nsym):
        bitstream = rng.random(bps*num_sc).round() # bits
        tones = modulation_mapper(bitstream, modorder) # convert from bits to modulated tones
        if en_tprecode: # transform precoding
            tones = fft.fft(tones)
        tones_nrb = np.zeros(nrb*12) + 1j*np.zeros(nrb*12) # available tones
        tones_nrb[start_sc:start_sc+num_sc] = tones # occupied tones
        tones_all = np.concatenate((np.zeros(round((nfft-nrb*12)/2)), tones_nrb, np.zeros(round((nfft-nrb*12)/2)))) # freq-domain zero-padding
        tones_all = np.concatenate((tones_all[round(nfft/2):], tones_all[0:round(nfft/2)])) # FFT shift prior to IFFT
        sym = fft.ifft(tones_all) # OFDM symbol
        sym_cp = np.concatenate((sym[nfft-ncp:], sym)) # add CP
        x_standard[sdx*sym_len:(sdx+1)*sym_len] = sym_cp # standard waveform
        
        # Apply WOLA
        sym_wola = np.concatenate((sym[round(nfft-ncp-wola_len/2):], sym, sym[0:round(wola_len/2)]))
        # sym_wola = np.multiply(sym_wola,b)
        sym_wola = sym_wola * b
        x[sdx*sym_len:(sdx+1)*sym_len+wola_len] = sym_wola + x[sdx*sym_len:(sdx+1)*sym_len+wola_len]
        
        # Save allocated tone indices for demodulation
        if sdx == 0:
            tone_idx = np.ones(nrb*12) # 1 = unallocated tones (IBE)
            tone_idx[start_sc:start_sc+num_sc] = 2 # 2 = allocated tones (EVM, equalizer)
            tone_idx = np.concatenate((np.zeros(round((nfft-nrb*12)/2)), tone_idx, np.zeros(round((nfft-nrb*12)/2)))) # 0 = zero-padded tones
            tone_idx = np.concatenate((tone_idx[round(nfft/2):], tone_idx[0:round(nfft/2)]))
    
    # Output dictionary for demodulation
    cfg = {
        "fs": fs,                               # sampling rate (MHz)
        "nfft": nfft,                           # FFT size
        "ncp": ncp,                             # cyclic prefix length (samples)
        "nrb": nrb,                             # number of resource blocks
        "nsym": nsym,                           # number of symbols
        "bw": bw,                               # BW (MHz)
        "scs": scs,                             # SCS (kHz)
        "num_sc": num_sc,                       # number of allocated subcarriers
        "start_sc": start_sc,                   # starting subcarrier
        "modorder": modorder,                   # modulation order
        "en_tprecode": en_tprecode,             # enable transform precoding
        "tone_idx": tone_idx,                   # allocated tone indices
        "wola_len": wola_len,                   # WOLA length (samples)
    }

    return (x, x_standard, cfg)

class OFDMWavGen:
    """
    OFDM waveform generator

    Constructor parameters
    ----------
    nsym: number of symbols to generate
    bw: signal bandwidth (MHz)
    scs: subcarrier spacing (kHz)
    num_sc: number of occupied subcarriers (leave as None to auto-calculate from start_sc)
    start_sc: starting subcarrier index
    modorder: modulation order (4, 16, 64, 256)
    en_tprecode: enable/disable transform precoding to switch b/w DFT-s-OFDM and CP-OFDM
    osr: oversampling ratio
    ncp: cyclic prefix length as a percentage of FFT size
    wola: WOLA length as a percentage of FFT size
    seed: RNG seed for repeatability
    power: power in a 50Ohm system (dBm) OR setpoint (-dBFS) if fixed-point
    bitwidth: if an integer, then it is the signed bitwidth for the fixed-point signal. Otherwise, generate a floating-point signal.

    Returns
    -------
    x: waveform with WOLA applied
    x_standard: waveform without WOLA applied
    cfg: dictionary that contains the waveform information for demodulation

    """

    def __init__(self, bw: int, scs: int, modorder: int, start_sc: int | None=None, num_sc: int | None=None, en_tprecode: bool=False, 
                 osr: int=1, ncp: float=7, wola: float=1, seed: int | None=None, power: float | None=None, bitwidth: bool | int = False):
        
        nrb = round(bw*5*15/scs) # max number of resource blocks
        nfft = (2**np.ceil(np.log2(nrb*12))).astype(np.uint).item() # matural FFT size (osr=1)
        nfft = nfft*osr # oversampling
        ncp = round(nfft*ncp/100) # number of samples for cyclic prefix
        sym_len = nfft+ncp # number of samples per symbol
        fs = nfft*scs/1000 # waveform sampling rate (MHz)
        bps = round(np.log2(modorder)) # bits per symbol
        wola_len = round(nfft*wola/100) # number of samples for WOLA window
        wola_len = wola_len + (wola_len % 2) # needs to be even (half outside of symbol, half inside of symbol)
        wola_win = OFDMWavGen.generate_wola_window(wola_len, sym_len) # WOLA window (number of samples = wola_len + sum_len)

        # Automatically calculate the allocated subcarriers
        if num_sc is None and start_sc is None: # if neither is provided, simply allocate all subcarriers
            start_sc = 0
            num_sc = nrb*12
        elif start_sc is None: # if num_sc is provided but not start_sc, center the allocation
            start_sc = round(nrb*12/2-num_sc/2)
        elif num_sc is None: # if start_sc is provided but not num_sc, allocate all subcarriers to the right of start_sc
            num_sc = nrb*12 - start_sc

        # Mask indicating which tones are allocated, unallocated, and zero-padded
        tone_idx = np.ones(nrb*12) # 1 = unallocated tones (IBE)
        tone_idx[start_sc:start_sc+num_sc] = 2 # 2 = allocated tones (EVM, equalizer)
        tone_idx = np.concatenate((np.zeros(round((nfft-nrb*12)/2)), tone_idx, np.zeros(round((nfft-nrb*12)/2)))) # 0 = zero-padded tones
        tone_idx = np.concatenate((tone_idx[round(nfft/2):], tone_idx[0:round(nfft/2)]))
        
        self.bw = bw
        self.scs = scs
        self.nrb = nrb
        self.nfft = nfft
        self.tone_idx = tone_idx
        self.fs = fs
        self.bps = bps
        self.modorder = modorder
        self.start_sc = start_sc
        self.num_sc = num_sc
        self.en_tprecode = en_tprecode
        self.osr = osr
        self.ncp = ncp
        self.sym_len = sym_len
        self.wola_len = wola_len
        self.wola_win = wola_win

        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        
        self.power = power
        self.bitwidth = bitwidth
    
    @staticmethod
    def generate_wola_window(wola_len, sym_len):
        """
        Generate a WOLA window

        Parameters
        ----------
        wola_len: length of the window in samples (must be even)
        sym_len: symbol length in samples

        Returns
        -------
        b: time-domain window for the entire symbol (sym_len + wola_len)
        
        """
        
        if wola_len > 0:
            b = signal.windows.hann(2*wola_len)
            b = np.concatenate((b[0:wola_len], np.ones(sym_len-wola_len), b[wola_len:]))
        else:
            b = np.ones(sym_len)
            
        return b
    
    def generate(self, nsym: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate an OFDM waveform

        Parameters
        ----------
        nsym: number of symbols
        seed: RNG seed. Provide an integer for repeatability (default is None).
        power: power in 50Ohm system (dBm). This is power for the complex baseband signal.

        Returns
        -------
        x: the waveform
        x_standard: a "standard" waveform without WOLA; use for calculating EVM
        
        """

        x_standard = np.zeros(nsym*self.sym_len) + 1j*np.zeros(nsym*self.sym_len) # no WOLA
        x = np.zeros(nsym*self.sym_len+self.wola_len) + 1j*np.zeros(nsym*self.sym_len+self.wola_len) # with WOLA
        for sdx in range(nsym):
            bitstream = self.rng.random(self.bps*self.num_sc).round() # bits
            tones = modulation_mapper(bitstream, self.modorder) # convert from bits to modulated tones
            if self.en_tprecode: # transform precoding
                tones = fft.fft(tones)
            tones_nrb = np.zeros(self.nrb*12) + 1j*np.zeros(self.nrb*12) # available tones
            tones_nrb[self.start_sc:self.start_sc+self.num_sc] = tones # occupied tones
            tones_all = np.concatenate((np.zeros(round((self.nfft-self.nrb*12)/2)), tones_nrb, np.zeros(round((self.nfft-self.nrb*12)/2)))) # freq-domain zero-padding
            tones_all = np.concatenate((tones_all[round(self.nfft/2):], tones_all[0:round(self.nfft/2)])) # FFT shift prior to IFFT
            sym = fft.ifft(tones_all) # OFDM symbol
            sym_cp = np.concatenate((sym[self.nfft-self.ncp:], sym)) # add CP
            x_standard[sdx*self.sym_len:(sdx+1)*self.sym_len] = sym_cp # standard waveform
            
            # Apply WOLA
            sym_wola = np.concatenate((sym[round(self.nfft-self.ncp-self.wola_len/2):], sym, sym[0:round(self.wola_len/2)]))
            # sym_wola = np.multiply(sym_wola,b)
            sym_wola = sym_wola * self.wola_win
            x[sdx*self.sym_len:(sdx+1)*self.sym_len+self.wola_len] = sym_wola + x[sdx*self.sym_len:(sdx+1)*self.sym_len+self.wola_len]
        
        if (self.bitwidth == False) and (self.power is not None):
            vrms = calc.dbm2v(self.power, "dBm")
            x = x/calc.rms(x)*vrms
        
        if self.bitwidth:
            if self.power is not None:
                x = x/calc.rms(x)*10**(self.power/20)
            else:
                x = x/np.abs(x).max()

            x = np.round(x*(2**(self.bitwidth-1)-1))

        return (x, x_standard)
    
    def calculate_evm(self, x: np.ndarray, y: np.ndarray, en_flat_eq: Literal["gain", "both", False]="gain", \
                      en_fd_eq: bool=False, **kwargs):
        """
        Calculates EVM (%) for OFDM waveform. Signals must be at baseband sampling rate and time-aligned.

        Parameters
        ----------
        x: reference
        y: signal under test
        en_flat_eq: enable a single-tap equalizer for the entire signal duration. Equalize only gain, gain+phase, or none.
        en_fd_eq: enable frequency-domain equalization - this estimates the gain and phase response of every subcarrier and equalizes the response.
        kwargs
            en_plot
            title

        Returns
        -------
        evm: error vector magnitude (%)
        

        If you assume all symbols are the same length, then you can also vectorize the symbol slicing
        And if you assume no dynamic power change, you can also vectorize the single-tap TD equalizer


        """

        en_plot = kwargs.get("en_plot", False)
        title = kwargs.get("title", "Constellation")

        x = x.copy()
        y = y.copy()
        if en_flat_eq != False:
            g_est = rf_est.est_single_tap_channel(x, y)
            if en_flat_eq == "gain":
                y = y/np.abs(g_est)
            elif en_flat_eq == "both":
                y = y/g_est
        
        # Take the FFT of each symbol to recover the modulated subcarriers
        # The output of this for loop is a matrix. Columns are subcarriers, rows are symbols.
        fft_start = round(self.ncp/2)
        tone_idx = self.tone_idx == 2 # allocated tones only
        n = min(len(x), len(y))-1-self.sym_len # last possible start index (sdx)
        for sdx in range(0, n, self.sym_len):
            sym_x = x[sdx:sdx+self.sym_len]
            sym_y = y[sdx:sdx+self.sym_len]
            
            # Remove cyclic prefix
            sym_x = sym_x[fft_start:fft_start+self.nfft]
            sym_y = sym_y[fft_start:fft_start+self.nfft]
            nshift = self.ncp-fft_start
            sym_x = np.roll(sym_x, -nshift)
            sym_y = np.roll(sym_y, -nshift)
            
            # FFT
            tones_x = fft.fft(sym_x)
            tones_y = fft.fft(sym_y)
            tones_x = tones_x[tone_idx]
            tones_y = tones_y[tone_idx]
            tones_x = tones_x.reshape(1, tones_x.size)
            tones_y = tones_y.reshape(1, tones_y.size)
            if sdx == 0:
                tones_xx = tones_x # nsym x ntone (occupied tones only)
                tones_yy = tones_y
            else:
                tones_xx = np.concatenate((tones_xx, tones_x))
                tones_yy = np.concatenate((tones_yy, tones_y))
        
        # Frequency-domain equalizer
        fd_eqs = np.ones(tones_x.size, dtype="complex") # Right now, this doesn't go anywhere, but it's essentially spectral flatness
        if en_fd_eq:
            # Loop over subcarriers and equalize
            for edx in range(0, fd_eqs.size):
                xx = tones_xx[:, edx]; yy = tones_yy[:, edx]
                
                # Standard definition
                fd_eq = np.vdot(xx, xx)/np.vdot(xx, yy)
                # fd_eq = np.vdot(yy,xx)/np.vdot(yy,yy) # This also works
                tones_yy[:, edx] = fd_eq*yy
                fd_eqs[edx] = fd_eq
                
        # Inverse transform precoding
        if self.en_tprecode:
            for sdx in range(tones_xx.shape[0]):
                tones_xx[sdx, :] = fft.ifft(tones_xx[sdx, :])
                tones_yy[sdx, :] = fft.ifft(tones_yy[sdx, :])
        
        # Calculate EVM
        err = tones_xx-tones_yy
        errv = err.reshape(1, err.size)
        tonesv = tones_xx.reshape(1, tones_xx.size)
        errp = np.vdot(errv, errv).real
        tonesp = np.vdot(tonesv, tonesv).real
        evm = np.sqrt(errp/tonesp)*100
        
        # Plot constellation
        if en_plot:
            tonesvx = tones_xx.reshape(1, tones_xx.size)
            tonesvy = tones_yy.reshape(1, tones_yy.size)
            c = calc.rms(tonesvx)
            tonesvx = tonesvx/c; tonesvy = tonesvy/c
            tonesvx = tonesvx[0]; tonesvy = tonesvy[0]
            
            fig = plt.figure()
            plt.plot(tonesvx.real, tonesvx.imag, 'x', markersize=5)
            plt.plot(tonesvy.real, tonesvy.imag, 'x', markersize=5)
            plt.title(title, {'fontsize':40})
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.grid()
        
        return evm

    def get_demod_params(self) -> dict:
        """
        Return a dictionary containing the waveform parameters needed for demodulation
        
        """
    
        # Output dictionary for demodulation
        cfg = {
            "fs": self.fs,                               # sampling rate (MHz)
            "nfft": self.nfft,                           # FFT size
            "ncp": self.ncp,                             # cyclic prefix length (samples)
            "nrb": self.nrb,                             # number of resource blocks
            # "nsym": self.nsym,                           # number of symbols
            "bw": self.bw,                               # BW (MHz)
            "scs": self.scs,                             # SCS (kHz)
            "num_sc": self.num_sc,                       # number of allocated subcarriers
            "start_sc": self.start_sc,                   # starting subcarrier
            "modorder": self.modorder,                   # modulation order
            "en_tprecode": self.en_tprecode,             # enable transform precoding
            "tone_idx": self.tone_idx,                   # allocated tone indices
            "wola_len": self.wola_len,                   # WOLA length (samples)
        }

        return cfg

def ofdm_evm_calculator(x: np.ndarray, y: np.ndarray, cfg: dict):
    """
    Calculates EVM (%) for OFDM waveform. Signals must be at baseband sampling rate and time-aligned.

    Parameters
    ----------
    x: reference
    y: signal under test
    cfg: dictionary containing demod params
        - 'en_fd_eq': enable FD equalization
        - 'en_plot'

    Returns
    -------
    evm: error vector magnitude (%)
    

    If you assume all symbols are the same length, then you can also vectorize the symbol slicing
    And if you assume no dynamic power change, you can also vectorize the single-tap TD equalizer

    The standard FD equalizer and my equalizer are the same - how?

    """
    
    # Convert from TD symbols to FD subcarriers
    nfft = cfg['nfft']
    ncp = cfg['ncp']
    sym_len = nfft+ncp
    fft_start = round(ncp/2)
    tone_idx = cfg['tone_idx']
    tone_idx = tone_idx == 2
    n = min(len(x), len(y))-1-sym_len # last possible start index (sdx)
    for sdx in range(0, n, sym_len):
        sym_x = x[sdx:sdx+sym_len]
        sym_y = y[sdx:sdx+sym_len]
        
        # Single-tap TD equalizer: Project x onto y
        # y'x / y'y * y
        sym_y = np.vdot(sym_y, sym_x)/np.vdot(sym_y, sym_y)*sym_y
        
        # Remove CP
        sym_x = sym_x[fft_start:fft_start+nfft]
        sym_y = sym_y[fft_start:fft_start+nfft]
        nshift = ncp-fft_start
        sym_x = np.roll(sym_x, -nshift)
        sym_y = np.roll(sym_y, -nshift)
        
        # FFT
        tones_x = fft.fft(sym_x)
        tones_y = fft.fft(sym_y)
        tones_x = tones_x[tone_idx]
        tones_y = tones_y[tone_idx]
        tones_x = tones_x.reshape(1,len(tones_x))
        tones_y = tones_y.reshape(1,len(tones_y))
        if sdx == 0:
            tones_xx = tones_x # nsym x ntone (occupied tones only)
            tones_yy = tones_y
        else:
            tones_xx = np.concatenate((tones_xx,tones_x))
            tones_yy = np.concatenate((tones_yy,tones_y))
    
    # FD equalizer
    fd_eqs = np.ones(tones_x.size) + 1j*np.zeros(tones_x.size)
    if cfg.get("en_fd_eq", False):
        for edx in range(0, len(fd_eqs)):
            xx = tones_xx[:, edx]; yy = tones_yy[:, edx]
            
            # Standard definition
            fd_eq = np.vdot(xx, xx)/np.vdot(xx, yy)
            tones_yy[:, edx] = fd_eq*yy
            fd_eqs[edx] = fd_eq
            
            # My definition - this also works
            # Project x onto y
            # y'x / y'y * y
            'fd_eq = np.vdot(yy,xx)/np.vdot(yy,yy)'
            
    # Inverse transform precoding
    if cfg['en_tprecode']:
        for sdx in range(0, tones_xx.shape[0]):
            tones_xx[sdx, :] = fft.ifft(tones_xx[sdx, :])
            tones_yy[sdx, :] = fft.ifft(tones_yy[sdx, :])
    
    # Calculate EVM
    err = tones_xx-tones_yy
    errv = err.reshape(1, err.size)
    tonesv = tones_xx.reshape(1, tones_xx.size)
    errp = np.vdot(errv, errv).real
    tonesp = np.vdot(tonesv, tonesv).real
    evm = np.sqrt(errp/tonesp)*100
    
    # Plot constellation
    if cfg.get("en_plot", False):
        titlestr = cfg.get("title", "Constellation")
        tonesvx = tones_xx.reshape(1, tones_xx.size)
        tonesvy = tones_yy.reshape(1, tones_yy.size)
        c = calc.rms(tonesvx)
        tonesvx = tonesvx/c; tonesvy = tonesvy/c
        tonesvx = tonesvx[0]; tonesvy = tonesvy[0]
        
        fig = plt.figure()
        plt.plot(tonesvx.real, tonesvx.imag, 'x', markersize=50)
        plt.plot(tonesvy.real, tonesvy.imag, 'x', markersize=25)
        plt.title(titlestr,{'fontsize':40})
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
    
    return evm