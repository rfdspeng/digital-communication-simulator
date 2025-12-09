# -*- coding: utf-8 -*-
"""
Created on Mon Nov 3 2025

Functions and classes for modeling generic digital hardware algorithms
and operations

@author: Ryan Tsai
"""

import numpy as np
from typing import Literal
import math
from scipy import signal

class RoundSat:
    def __init__(self):
        pass

def upsample(x: np.ndarray, M: int | float) -> np.ndarray:
    """
    Integer upsampling

    Arguments
        x: numpy array
        M: upsampling ratio

    """
    assert x.ndim == 1, "Input vector should be 1-dimensional"

    M = int(M)
    y = np.zeros(x.size*M, dtype=x.dtype)
    y[::M] = x
    
    return y

def downsample(x: np.ndarray, M: int | float) -> np.ndarray:
    """
    Integer downsampling
    
    Arguments
        x: numpy array
        M: downsampling ratio

    """

    assert x.ndim == 1, "Input vector should be 1-dimensional"

    return x[::M].copy()

def polyphase_decomposition(b, R, mode: Literal["conventional", "symmetric"]="conventional"):
    """
    Decompose prototype filter b into R branches.
    "conventional": conventional decomposition
    "symmetric": make all branches symmetric or antisymmetric. See "Restoring Coefficient Symmetry in Polyphase Implementation of Linear-Phase FIR Filters".
    
    Assume even-order filter (b is odd).
    """

    assert b.size % 2 == 1, "Please provide an even-order filter"
    b = b.copy().astype("float")

    if mode == "conventional":
        branch_len = math.ceil(b.size/R)
        b_pp = np.zeros((R, branch_len))
        b = np.concatenate((b, np.zeros(b_pp.size-len(b))))
        for r in range(R):
            b_pp[r, :] = b[r::R]

    elif mode == "symmetric":
        N0 = (b.size-1)/2
        p = math.ceil(N0/R)
        N = p*R
        b = np.concatenate((np.zeros(round(N-N0)), b, np.zeros(round(N-N0))))

        # Conventional polyphase decomposition
        b_pp = np.zeros((R, 2*p+1), dtype="float")
        b_pp[0, :] = b[::R]
        for r in range(1, R):
            b_pp[r, :-1] = b[r::R]
        
        if R == 2:
            return b_pp

        # Transformation to symmetric polyphase branches
        b_pp_sym = np.zeros((R, 2*p+1), dtype="float")
        b_pp_sym[0, :] = b_pp[0, :]
        if R % 2 == 0: # R is even and > 2
            b_pp_sym[round(R/2), :] = b_pp[round(R/2), :]
            r_tr_max = round(R/2-1)

        else: # R is odd
            r_tr_max = math.floor(R/2)
        
        # r_tr_max is the index of the last "first-half" branch to transform
        for r in range(1, r_tr_max+1):
            b_pp_sym[r, :] = 1/2*(b_pp[r, :] + b_pp[R-r, :])
            b_pp_sym[R-r, :] = 1/2*(b_pp[r, :] - b_pp[R-r, :])
        
        return b_pp_sym
    
    return b_pp

class PolyphaseDownsampler:
    def __init__(self, b, decomp_type: Literal["conventional", "symmetric"]="conventional", sim_type: Literal["vectorized", "hardware"]="vectorized"):
        """
        b = polyphase coefficients. R x 2p+1. R = rate change. 2p+1 = number of coefficients in branch 0. All other branches have 2p.
        
        """
        self.b = b
        self.decomp_type = decomp_type
        self.sim_type = sim_type

        self.R_ = b.shape[0]
        if self.decomp_type == "symmetric":
            self.r_symm_ = [0]
            if self.R_ % 2 == 0:
                self.r_symm_.append(round(self.R_/2))
                self.r_tr_max_ = round(self.R_/2-1)
            else:
                self.r_tr_max_ = math.floor(self.R_/2)

    
    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.sim_type == "vectorized":
            if self.decomp_type == "conventional":
                y = np.zeros((self.R_, math.ceil(x.size/self.R_)), dtype="complex")
                for r in range(self.R_):
                    x_br = downsample(x[r:], self.R_)
                    y_br = signal.lfilter(self.b[r, :], 1, x_br)

                    # y[r, :y_br.size] = y_br

                    if r > 0:
                        y[r, 1:y_br.size+1] = y_br[:y.shape[1]-1]
                    elif r == 0:
                        y[r, :y_br.size] = y_br

            elif self.decomp_type == "symmetric":
                y = np.zeros((self.R_, math.ceil(x.size/self.R_)), dtype="complex")
                for r in self.r_symm_:
                    x_br = downsample(x[r:], self.R_)
                    y_br = signal.lfilter(self.b[r, :], 1, x_br)
                    y[r, :y_br.size] = y_br

                for r in range(1, self.r_tr_max_+1):
                    x_br = downsample(x[r:], self.R_)
                    y_br = signal.lfilter(self.b[r, :] + self.b[self.R_-r, :], 1, x_br)
                    y[r, :y_br.size] = y_br
                    x_br = downsample(x[self.R_-r:], self.R_)
                    y_br = signal.lfilter(self.b[r, :] - self.b[self.R_-r, :], 1, x_br)
                    y[self.R_-r, :y_br.size] = y_br
                
            y = y.sum(axis=0).squeeze()

        elif self.sim_type == "hardware":
            pass

        return y

def polyphase_downsampler(x: np.ndarray, b, R, frac_bits: int | float | bool = False):
    """
    Description
    -----------
    Implements cycle-based polyphase downsampler

    Parameters
    ----------
    x : input signal
    b : prototype filter coefficients
    frac_bits : fractional bitwidth for normalization (0/False means floating point)
    R : integer downsampling ratio

    Returns
    -------
    y : output signal

    """
    
    # Generate branches
    branch_len = math.ceil(len(b)/R)
    b_pp = np.zeros((R, branch_len))
    gd = int((len(b)-1)/2)
    b = np.concatenate((b, np.zeros(b_pp.size-len(b)+1)))
    for branch in range(R):
        b_pp[branch,:] = b[branch:-1:R]
    
    reg_in = np.zeros(branch_len*R) + 1j*np.zeros(branch_len*R) # input tapped delay line
    
    y = np.zeros(math.floor(len(x)/R)) + 1j*np.zeros(math.floor(len(x)/R))
    
    # Zero pad the input signal
    x = np.concatenate((np.zeros(len(reg_in)-1), x, np.zeros(gd)))
    
    branch = 0 # branch index
    y_br = np.zeros(R) + 1j*np.zeros(R) # branch outputs
    ydx = 0 # output index
    for x_end in range(len(reg_in)-1, len(x)):
        # Update input tapped delay line
        reg_in[:] = np.flip(x[x_end-len(reg_in)+1:x_end+1])
        x_tapped = reg_in[0:-1:R]
        
        b_br = b_pp[branch,:] # branch coefficients
        
        # Compute branch output
        conv_out = sum(b_br*x_tapped)
        #if frac_bits > 0:
        #    conv_out = round(conv_out/2**frac_bits)
    
        y_br[branch] = conv_out
        
        # Compute output sample
        if branch == 0:
            y[ydx] = sum(y_br)
            if frac_bits > 0:
                # y[ydx] = round(y[ydx]/2**frac_bits)
                y[ydx] = (y[ydx]/2**frac_bits).round()
            ydx += 1
        
        # Update branch
        if branch == 0:
            branch = R-1
        else:
            branch -= 1
        """
        if branch == 2:
            branch = 0
        else:
            branch += 1
        """
        
        if ydx >= len(y):
            break

    return y

class CORDIC:
    """
    CORDIC
        
    """

    def __init__(self, N, mode: Literal["vectoring", "rotation"], sim_type: Literal["vectorized", "hardware"]="vectorized", theta: np.ndarray | None=None):
        """
        N = number of iterations
        mode = vectoring or rotation
        theta = for rotation mode only.
            For polar-rect conversion, set to None. Each sample is (env, ph), and env + 0j is rotated by exp(1j*ph)
            For phase/frequency shift, set to desired phase shift per sample (constant for phase shift, accumulated for frequency shift).
                Each sample is (env, ph), but this is actually rectangular coordinates (I, Q).

        """

        self.N = N
        self.mode = mode
        self.sim_type = sim_type
        self.theta = theta

        self.K_ = np.prod(1/np.sqrt(1+2**(-2*np.arange(N, dtype="float"))))
        self.theta_i_ = CORDIC.get_rotation_angles(N)

        self.rot_mat_p_ = [np.array([[1, -1/2**idx],[1/2**idx, 1]]) for idx in range(N)]
        self.rot_mat_m_ = [np.array([[1, 1/2**idx],[-1/2**idx, 1]]) for idx in range(N)]
    
    @staticmethod
    def get_rotation_angles(N):
        """
        Get the angles of rotation for each iteration (radians)

        """
        return np.arctan(1/2**(np.arange(N)))
    
    def transform(self, I: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        In vectoring mode, the input are the rectangular representation I and Q
        In rotation mode, the inputs are the polar representation env and phase
        
        """

        if self.mode == "vectoring":
            I = I.copy()
            Q = Q.copy()
            self.theta_acc_ = np.zeros(I.size)

            self.theta_acc_[I < 0] = np.pi
            Q[I < 0] = -1*Q[I < 0]
            I[I < 0] = -1*I[I < 0]

            I = I.reshape((1, I.size))
            Q = Q.reshape((1, Q.size))
            IQ = np.vstack((I, Q))
            env, ph = self.vectoring(IQ)

            return (env*self.K_, ph)

        elif self.mode == "rotation":
            # if self.theta is None:
            #     # Polar-to-rectangular conversion
            #     env = I.copy()
            #     ph = Q.copy()
                
            #     ph = np.mod(ph, 2*np.pi) # wrap to 0 to 2*pi
            #     ph[ph >= np.pi] = ph[ph >= np.pi] - 2*np.pi # wrap to -pi, pi
            #     self.final_multiply_ = np.ones(ph.size)
            #     self.final_multiply_[np.logical_or(ph < -np.pi/2, ph > np.pi/2)] = -1
            #     ph[ph < -np.pi/2] = ph[ph < -np.pi/2] + np.pi
            #     ph[ph > np.pi/2] = ph[ph > np.pi/2] - np.pi
            #     self.err_ = ph

            #     I = env.reshape((1, env.size))
            #     Q = np.zeros_like(I)
            #     IQ = np.vstack((I, Q))
            # else:
            #     I = I.copy()
            #     Q = Q.copy()
            #     ph = self.theta
            
            if self.theta is not None:
                # Phase or frequency shift
                I = I.copy()
                Q = Q.copy()
                ph = self.theta.copy()
            else:
                # Polar-to-rectangular conversion
                env = I.copy()
                ph = Q.copy()
            
            ph = np.mod(ph, 2*np.pi) # wrap to 0 to 2*pi
            ph[ph >= np.pi] = ph[ph >= np.pi] - 2*np.pi # wrap to -pi, pi
            self.final_multiply_ = np.ones(ph.size)
            self.final_multiply_[np.logical_or(ph < -np.pi/2, ph > np.pi/2)] = -1
            ph[ph < -np.pi/2] = ph[ph < -np.pi/2] + np.pi
            ph[ph > np.pi/2] = ph[ph > np.pi/2] - np.pi
            self.err_ = ph

            if self.theta is None:
                I = env.reshape((1, env.size))
                Q = np.zeros_like(I)
            
            IQ = np.vstack((I, Q))
            I, Q = self.rotation(IQ)

            return (I*self.K_, Q*self.K_)

    def vectoring(self, IQ):
        """
        IQ is a 2 x nsamp matrix
        First row is I
        Second row is Q

        """

        IQ = IQ.copy()
        if self.sim_type == "vectorized":
            for i in range(self.N):
                y_sign = np.sign(IQ[1,:])
                IQp = self.rot_mat_p_[i] @ IQ
                IQm = self.rot_mat_m_[i] @ IQ
                IQ[:, y_sign == -1] = IQp[:, y_sign == -1]
                IQ[:, y_sign == +1] = IQm[:, y_sign == +1]
                # IQ[:, y_sign == 0] Do nothing, you've converged

                self.theta_acc_ = self.theta_acc_ + -1*y_sign*self.theta_i_[i]
            
            env = IQ[0, :]
            ph = -self.theta_acc_.copy()
        elif self.sim_type == "hardware":
            pass
        
        return (env, ph)
    
    def rotation(self, IQ):
        """
        IQ is a 2 x nsamp matrix
        First row is I (starting value is env)
        Second row is Q (starting value is 0)

        """

        IQ = IQ.copy()
        if self.sim_type == "vectorized":
            for i in range(self.N):
                err_sign = np.sign(self.err_)
                IQp = self.rot_mat_p_[i] @ IQ
                IQm = self.rot_mat_m_[i] @ IQ
                IQ[:, err_sign == 1] = IQp[:, err_sign == 1]
                IQ[:, err_sign == -1] = IQm[:, err_sign == -1]
                # IQ[:, err_sign == 0] Do nothing, you've converged

                self.err_ = self.err_ - err_sign*self.theta_i_[i]
                        
            I = IQ[0,:]*self.final_multiply_
            Q = IQ[1,:]*self.final_multiply_
        elif self.sim_type == "hardware":
            pass

        return (I, Q)

class SPDFT:
    """
    Single-point DFT for estimating amplitude/phase at a single frequency

    """

    def __init__(self, w0, sim_type: Literal["vectorized", "hardware"]="vectorized"):
        """
        w0 = digital frequency (radians/sample)
        
        """

        self.w0 = w0
        self.sim_type = sim_type
    
    def transform(self, x: np.ndarray):
        if self.sim_type == "vectorized":
            return (x*np.exp(-1j*self.w0*np.arange(x.size, dtype="float"))).sum()
        
        elif self.sim_type == "hardware":
            pass