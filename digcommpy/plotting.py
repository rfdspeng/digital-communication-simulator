# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 08:55:00 2025

Functions for plotting

@author: Ryan Tsai
"""

from typing import Union
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from scipy import fft, signal
import warnings

def plot_freqs(bk: Union[ArrayLike, int, float], ak: Union[ArrayLike, int, float], **kwargs):
    """
    Can be used to plot FFT of signals (ak = 1) or LTI systems
    Plot CTFT of CT signals (ak = 1) or LTI systems
    
    """

    worN = kwargs.get("worN", 4096)
    angular_freq = kwargs.get("angular_freq", False)
    title = kwargs.get("title", "Freqs Plot")

    w, h = signal.freqs(bk, ak, worN=worN)
    if not angular_freq:
        w = w/2/np.pi

    h_re_imag = np.concatenate((h.real, h.imag))
    ymin = h_re_imag.min()
    ymax = h_re_imag.max()
    ymin2 = (ymin + ymax)/2 - (ymax - ymin)*0.55
    ymax2 = (ymin + ymax)/2 + (ymax - ymin)*0.55

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, dpi=150, figsize=(12, 6))
    axs[0, 0].semilogx(w, 20*np.log10(np.abs(h)))
    axs[0, 0].set_ylabel("Magnitude (dB)")
    axs[0, 0].grid()
    axs[0, 0].set_xlim(left=w.min(), right=w.max())
    axs[0, 0].tick_params(labelbottom=True)
    axs[1, 0].semilogx(w, np.angle(h))
    axs[1, 0].grid()
    axs[1, 0].set_ylabel("Phase (Rad)")
    if angular_freq:
        axs[1, 0].set_xlabel("Frequency (Rad/Second)")
    else:
        axs[1, 0].set_xlabel("Frequency (Hz)")
    axs[0, 1].plot(w, h.real)
    axs[0, 1].set_ylabel("Real")
    axs[0, 1].set_ylim(bottom=ymin2, top=ymax2)
    axs[0, 1].grid()
    axs[0, 1].tick_params(labelbottom=True)
    axs[1, 1].plot(w, h.imag)
    axs[1, 1].grid()
    axs[1, 1].set_ylabel("Imaginary")
    if angular_freq:
        axs[1, 1].set_xlabel("Frequency (Rad/Second)")
    else:
        axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylim(bottom=ymin2, top=ymax2)
    fig.suptitle(title)

    return (fig, axs)

def plot_freqz(bk: Union[ArrayLike, int, float], ak: Union[ArrayLike, int, float], **kwargs):
    """
    Can be used to plot FFT of signals (ak = 1) or LTI systems
    
    """

    worN = kwargs.get("worN", 4096)
    whole = kwargs.get("whole", True)
    yscale = kwargs.get("yscale", "decibel") # 'decibel', 'linear'
    title = kwargs.get("title", "Freqz Plot")

    w, h = signal.freqz(bk, ak, worN=worN, whole=whole)
    w = fft.fftshift(w)
    w[w >= np.pi] = w[w >= np.pi] - 2*np.pi
    h = fft.fftshift(h)

    h_re_imag = np.concatenate((h.real, h.imag))
    ymin = h_re_imag.min()
    ymax = h_re_imag.max()
    ymin2 = (ymin + ymax)/2 - (ymax - ymin)*0.55
    ymax2 = (ymin + ymax)/2 + (ymax - ymin)*0.55

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, dpi=150, figsize=(9, 6))
    if yscale == "decibel":
        axs[0, 0].plot(w, 20*np.log10(np.abs(h)))
        axs[0, 0].set_ylabel("Magnitude (dB)")
    elif yscale == "linear":
        axs[0, 0].plot(w, np.abs(h))
        axs[0, 0].set_ylabel("Magnitude (Linear)")
    axs[0, 0].grid()
    axs[0, 0].set_xlim(left=-np.pi, right=np.pi)
    axs[0, 0].tick_params(labelbottom=True)
    axs[1, 0].plot(w, np.angle(h))
    axs[1, 0].grid()
    axs[1, 0].set_ylabel("Phase (Rad)")
    axs[1, 0].set_xlabel("Frequency (Rad/Sample)")
    axs[0, 1].plot(w, h.real)
    axs[0, 1].set_ylabel("Real")
    axs[0, 1].set_ylim(bottom=ymin2, top=ymax2)
    axs[0, 1].grid()
    axs[0, 1].tick_params(labelbottom=True)
    axs[1, 1].plot(w, h.imag)
    axs[1, 1].grid()
    axs[1, 1].set_ylabel("Imaginary")
    axs[1, 1].set_xlabel("Frequency (Rad/Sample)")
    axs[1, 1].set_ylim(bottom=ymin2, top=ymax2)
    fig.suptitle(title)

    return (fig, axs)

def compare_freqz(bk: list[Union[ArrayLike, int, float]], ak: list[Union[ArrayLike, int, float]], **kwargs):
    """
    Can be used to plot FFT of signals (ak = 1) or LTI systems
    
    """

    assert len(bk) == len(ak), "bk and ak coefficient lists must be the same length"

    worN = kwargs.get("worN", [4096]*len(bk))
    whole = kwargs.get("whole", [True]*len(bk))
    labels = kwargs.get("labels", [str(i) for i in range(len(bk))])
    yscale = kwargs.get("yscale", "decibel") # 'decibel', 'linear'
    title = kwargs.get("title", "Freqz Plot")

    fig, axs = plt.subplots(dpi=150)
    for idx in range(len(bk)):
        w, h = signal.freqz(bk[idx], ak[idx], worN=worN[idx], whole=whole[idx])

        if whole[idx]:
            w = fft.fftshift(w)
            w[w >= np.pi] = w[w >= np.pi] - 2*np.pi
            h = fft.fftshift(h)

        if yscale == 'decibel':
            axs.plot(w, 20*np.log10(np.abs(h)), label=labels[idx])
        elif yscale == 'linear':
            axs.plot(w, np.abs(h), label=labels[idx])
    
    axs.set_xlim(left=w.min(), right=w.max())
    axs.set_xlabel("Frequency (Rad/Sample)")
    if yscale == 'decibel':
        axs.set_ylabel("Magnitude (dB)")
    elif yscale == 'linear':
        axs.set_ylabel("Magnitude (Linear)")
    axs.grid()
    axs.legend()
    axs.set_title(title)

    return (fig, axs)

def plot_ct_sig(t: np.ndarray, x: np.ndarray, labels: list[str] | None=None, title: str | None=None):
    """
    To plot multiple signals, let t and x have shape nsig * nsamp
    
    """
    t = t.copy()
    x = x.copy()

    if t.ndim == 1:
        t = t[np.newaxis, :]
    if x.ndim == 1:
        x = x[np.newaxis, :]
    
    assert t.shape == x.shape

    labels = labels.copy() if labels is not None else [f"x{i}" for i in range(x.shape[0])]
    assert x.shape[0] == len(labels)

    fig, ax = plt.subplots(dpi=150)
    for idx in range(x.shape[0]):
        ax.plot(t[idx, :], x[idx, :], label=labels[idx]) #, linewidth = 10/2**(idx+1))

    ax.set_xlim(left=0, right=t.max())
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal Amplitude")
    ax.grid()
    ax.legend(fontsize=8)
    if title is None:
        title = "CT Signals"
    fig.suptitle(title)

    return (fig, ax)

def plot_ctfs(f: np.ndarray, Xk: np.ndarray, title: str | None=None):
    f = f.copy()
    Xk = Xk.copy()

    fig, axs = plt.subplots(nrows=2, sharex=True, dpi=150)
    axs[0].stem(f, np.abs(Xk))
    axs[0].set_ylabel("|Xk|")
    axs[0].tick_params(labelbottom=True)
    axs[0].grid()
    axs[1].stem(f, np.angle(Xk))
    axs[1].set_xlim(left=f.min(), right=f.max())
    axs[1].set_ylabel("Angle Xk (rad)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].grid()
    if title is None:
        title = "CT Fourier Series"
    fig.suptitle(title)

    return (fig, axs)