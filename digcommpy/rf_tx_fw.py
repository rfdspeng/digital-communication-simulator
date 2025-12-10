# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 2025

Functions and classes for modeling Tx DSP and FW algorithms

@author: Ryan Tsai
"""

import numpy as np

def spdft():
    pass

def tx_iq_mm_est():
    pass

def gmp_kernel_matrix(x: np.ndarray, ktups: list[tuple]) -> tuple[np.ndarray, list[str]]:
    """
    Generate kernel matrix for DPD training

    ktups is a list of tuples. Each tuple consists of (identifier string, params)
    The identifier string is 'GMP', 'DDR', etc. The params depend on the kernel type.

    """

    x = x.reshape(x.size, 1, copy=True)
    x_abs = abs(x)
    
    kmat = np.empty(0)
    kstr = []
    # for kdx in range(len(ktups)):
    for kdx, ktup in enumerate(ktups):
        # ktup = ktups[kdx]
        ktype = ktup[0]
    
        if ktype == 'GMP':
            p = ktup[1] # Total order
            m = ktup[2] # Complex delay
            l = ktup[3] # Envelope delay (relative to complex delay)
            
            # Complex term
            if m == 0:
                x_iq = x
            elif m > 0:
                x_iq = np.vstack( (np.zeros((m,1)), x[0:-m]) )
            elif m < 0:
                x_iq = np.vstack( (x[-m:], np.zeros((-m,1))) )
            
            # Envelope term
            if m+l == 0:
                x_env = x_abs
            elif m+l > 0:
                x_env = np.vstack( (np.zeros((m+l,1)), x_abs[0:-(m+l)]) )
            elif m+l < 0:
                x_env = np.vstack( (x_abs[-(m+l):], np.zeros((-(m+l),1))) )
                
            kernel = x_iq*x_env**(p-1)
            kid = 'x' + str(m) if p == 1 else 'x' + str(m) + '|x' + str(l) + '|^' + str(p-1)
        
        kmat = kernel if kdx == 0 else np.hstack((kmat, kernel))
        kstr.append(kid) 
        
    return (kmat, kstr)

def ila_dpd_training(x: np.ndarray, y: np.ndarray, kernels: list[tuple] | None=None):
    """
    DPD training using the inverse model
    
    Parameters
    ----------
    x: PA input
    y: PA output
    kernels: list of tuples that define the DPD kernels
        - kernel[0] = kernel type (only 'GMP')
        - kernel[1] = order of the term (complex + envelope)
        - kernel[2] = delay of the complex term
        - kernel[3] = delay of the envelope term relative to the complex term

    """

    x = x.reshape(x.size, 1, copy=True)
    y = y.reshape(y.size, 1, copy=True)

    if kernels is None:
        # Memoryless DPD
        kernels = [("GMP", 1, 0, 0),
                   ("GMP", 3, 0, 0),
                   ("GMP", 5, 0, 0),
                   ("GMP", 7, 0, 0),
                   ("GMP", 9, 0, 0)]

    K, Kstr = gmp_kernel_matrix(y, kernels)

    # KHK = K.conjugate().transpose() @ K
    # L = np.linalg.cholesky(KHK)
    c = np.linalg.pinv(K) @ x

    env2 = x*x.conjugate()

    # Hardcoded
    # Returns the predistorted signal - TBD: return LUT instead
    x_dpd = x*(c[0] + c[1]*env2 + c[2]*env2**2 + c[3]*env2**3 + c[4]*env2**4)

    return (x_dpd.squeeze(), Kstr)