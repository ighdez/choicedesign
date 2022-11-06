"""Optimisation criteria"""

# Import modules
import numpy as np
from choicedesign.utils import _dummygen

# D-error function
def _derr(imat, design: np.ndarray, levs: list, coding: list, pars: list, natts: int, nalts: int, ncs: int, optout: bool, asc: dict):
    """D-error of an information matrix"""
    # Get information matrix 
    im = imat(design, levs, coding, pars, natts, nalts, ncs, optout, asc)

    # Calculate D-error
    if np.linalg.det(im) != 0:
        vce = np.linalg.solve(im,np.eye(im.shape[0]))

        if asc is not None:
            vce = vce[len(asc):,len(asc):]

        detvce = np.linalg.det(vce)
        dr = detvce**(1/vce.shape[0])

    else:
        dr = np.inf

    return dr

# Information matrix of MNL model
def _imat_mnl(design: np.ndarray, levs: list, coding: list, pars: list, natts: int, nalts: int, ncs: int, optout: bool, asc: dict):
    """Information matrix of the MNL model"""
    # Populate the coded matrix
    coded_matrix = []

    for k in range(len(coding)):
        if coding[k] == 'dummy':
            dd = _dummygen(design[:,k],levs[k])
            coded_matrix.append(dd)
        else:
            coded_matrix.append(design[:,k][:,np.newaxis])
    
    coded_matrix = np.concatenate(coded_matrix,axis=1)

    # If optout alternative is present, add to coded matrix
    if optout:
        len_asc = (len(asc) if asc is not None else 0)
        coded_matrix = np.c_[coded_matrix,np.zeros((ncs,int(len(pars)/(nalts+optout))-len_asc))]
        
    # Calculate utility of each alternative
    v = coded_matrix * pars
    v.shape = (ncs, nalts+optout, int(len(pars)/(nalts+optout)))
    v = v.sum(axis=2)

    # Calculate Probability
    ev = np.exp(v)
    sev = ev.sum(axis=1,keepdims=True)
    p = (ev/sev).flatten()

    # Calculate Information Matrix
    coded_matrix.shape = (ncs * (nalts+optout), int(len(pars)/(nalts+optout)))
    
    # Calculate Information Matrix
    ia = np.diag(p) @ coded_matrix
    ia = ia.T @ coded_matrix
    
    ib = np.diag(p) @ coded_matrix
    ib.shape = (ncs,nalts+optout,int(len(pars)/(nalts+optout)))
    ib = np.sum(ib,axis=1)
    ib = ib.T @ ib
    
    im = ia - ib

    # Return information matrix
    return im

# Utility balance function
def _utility_balance_mnl(design: np.ndarray, levs: list, coding: list, pars: list, natts: int, nalts: int, ncs: int, optout: bool, asc: dict):
    """Utility balance ratio of the MNL model"""
    # Populate the coded matrix
    coded_matrix = []

    for k in range(len(coding)):
        if coding[k] == 'dummy':
            dd = _dummygen(design[:,k],levs[k])
            coded_matrix.append(dd)
        else:
            coded_matrix.append(design[:,k][:,np.newaxis])
    
    coded_matrix = np.concatenate(coded_matrix,axis=1)

    # If optout alternative is present, add to coded matrix
    if optout:
        len_asc = (len(asc) if asc is not None else 0)
        coded_matrix = np.c_[coded_matrix,np.zeros((ncs,int(len(pars)/(nalts+optout))-len_asc))]
        
    # Calculate utility of each alternative
    v = coded_matrix * pars
    v.shape = (ncs, nalts+optout, int(len(pars)/(nalts+optout)))
    v = v.sum(axis=2)

    # Calculate Probability
    ev = np.exp(v)
    sev = ev.sum(axis=1,keepdims=True)
    p = (ev/sev)

    # Calculate utility balance
    B = (p/(1/(nalts+optout))).copy()
    B = np.prod(B,axis=1)*100
    B = np.sum(B)/ncs

    return B