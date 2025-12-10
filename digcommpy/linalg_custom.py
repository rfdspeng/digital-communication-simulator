# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 09:00:29 2023

Custom linear algebra functions

@author: Ryan Tsai
"""

import numpy as np
from numpy import linalg

def identity_matrix(n):
    """
    Generate nxn identity matrix
    
    """
    I = np.zeros((n,n)) + 0j*np.zeros((n,n))
    for idx in range(n):
        I[idx,idx] = 1
    
    return I

def rank(A,tol=None):
    """
    Computes the rank of A using SVD, where A is an m x n matrix
    
    If tol is None, then default tolerance is used: (largest singular value) * max(m,n) * eps
    
    Otherwise, tol is a positive number in dB

    """
    
    AHA = A.T.conj() @ A
    eigvals = linalg.eigvalsh(AHA)
    svals = np.sqrt(eigvals)
    if tol == None:
        thr = max(svals)*max(A.shape)*np.finfo(max(svals)).eps
    else:
        thr = max(svals)*10**(-tol/20)
        
    rk = sum(svals >= thr)
    return rk

def eliminate(A,prune_thr_db=999):
    """
    A is a square matrix
    prune_thr_db is the pruning threshold in dB (a positive value)
    When a pivot is less than (first pivot)*10^(-prune_thr_db/20), zero out the row and column and move to the next one
    
    Returns E, U, and a vector that contains the indices (starting from 0) of the pruned rows and columns
    E is the elimination matrix
    U is the upper triangular matrix after elimination (U = EA)
    
    """
    
    if not(len(A.shape) == 2):
        raise Exception("eliminate: A must be 2-dimensional")
    elif not(A.shape[0] == A.shape[1]):
        raise Exception("eliminate: A must be a square matrix")
    elif prune_thr_db < 0:
        raise Exception("eliminate: prune_thr_db must be positive")
        
    ndim = A.shape[0] # Matrix size
    E = identity_matrix(ndim)
    U = A.copy()
    elim_idx = []
    for idx in range(ndim):
        pivot = U[idx,idx]
        if idx == 0:
            first_pivot = abs(pivot)
        
        if abs(pivot) <= first_pivot*10**(-prune_thr_db/20):
            # Pruning condition
            U[idx,:] = 0
            U[:,idx] = 0
            elim_idx.append(idx)
        else:
            # Elimination
            Ei = identity_matrix(ndim)
            for jdx in range(idx+1,ndim):
                c = -U[jdx,idx]/pivot
                U[jdx,:] = c*U[idx,:] + U[jdx,:]
                Ei[jdx,idx] = c
                
            E = Ei @ E
    
    E[:,elim_idx] = 0
    E[elim_idx,:] = 0       
    return (E,U,elim_idx)

def lu(A,prune_thr_db=999):
    """
    A is a square matrix
    prune_thr_db is the pruning threshold in dB (a positive value)
    When a pivot is less than (first pivot)*10^(-prune_thr_db/20), zero out the row and column and move to the next one
    
    Returns L, U, and a vector that contains the indices (starting from 0) of the pruned rows and columns
    
    """
    
    if not(len(A.shape) == 2):
        raise Exception("lu: A must be 2-dimensional")
    elif not(A.shape[0] == A.shape[1]):
        raise Exception("lu: A must be a square matrix")
    elif prune_thr_db < 0:
        raise Exception("lu: prune_thr_db must be positive")
        
    [E,U,elim_idx] = eliminate(A,prune_thr_db=999)
    
    L = np.linalg.inv(E) # Write function to invert triangular matrix
    L[:,elim_idx] = 0
    L[elim_idx,:] = 0
    return (L,U,elim_idx)

def chol(A,prune_thr_db=999):
    """
    A is a Hermitian matrix
    prune_thr_db is the pruning threshold in dB (a positive value)
    When a pivot is less than (first pivot)*10^(-prune_thr_db/20), zero out the row and column and move to the next one
    
    Returns L and a vector that contains the indices (starting from 0) of the pruned rows and columns
    
    """
    
    
    
    
    
    
    
    
    return 1

def solve_ls(A,b,decomp_option="eliminate",prune_thr_db=999):
    """
    Solve Ax = b
    
    decomp_option: Choose the algorithm to use for decomposition of A
    
    """
    
    if b.ndim == 1:
        b = b.reshape((len(b),1))
    
    if not(len(A.shape) == 2):
        raise Exception("solve_ls: A must be 2-dimensional")
    elif not(len(b.shape) == 2):
        raise Exception("solve_ls: b must be 2-dimensional")
    elif not(b.shape[1] == 1):
        raise Exception("solve_ls: b must have 1 column")
    elif not(A.shape[0] == b.shape[0]):
        raise Exception("solve_ls: A and b must have the same number of rows")
    
    # Ax = b
    # AHAx = AHb
    AHA = A.T.conj() @ A
    AHb = A.T.conj() @ b
    
    if decomp_option == "eliminate":
        # AHAx = AHb
        # EAHAx = EAHb -> EAHA = U
        # Ux = EAHb
        [E,U,elim_idx] = eliminate(AHA,prune_thr_db=prune_thr_db)
        EAHb = E @ AHb
        
        # Solve Ux = EAHb for x
        x = np.zeros((U.shape[0],1)) + 0j*np.zeros((U.shape[0],1))
        for idx in range(U.shape[0]-1,-1,-1):
            if not(idx in elim_idx):
                x[idx] = (EAHb[idx] - (U[idx,:] @ x))/U[idx,idx]
            else:
                x[idx] = 0
        
    elif decomp_option == "lu":
        # LUx = AHb
        [L,U,elim_idx] = lu(AHA,prune_thr_db=prune_thr_db)
        
        # Solve Ly = AHb for y, where y = Ux
        y = np.zeros((L.shape[0],1)) + 0j*np.zeros((L.shape[0],1))
        for idx in range(L.shape[0]):
            if not(idx in elim_idx):
                y[idx] = (AHb[idx] - (L[idx,:] @ y))/L[idx,idx]
            else:
                y[idx] = 0
        
        # Solve Ux = y for x
        x = np.zeros((L.shape[0],1)) + 0j*np.zeros((L.shape[0],1))
        for idx in range(U.shape[0]-1,-1,-1):
            if not(idx in elim_idx):
                x[idx] = (y[idx] - (U[idx,:] @ x))/U[idx,idx]
            else:
                x[idx] = 0
    
    return (x.reshape((len(x))),elim_idx)