"""Optimisation algorithms"""

# Load modules
import datetime
import time
import numpy as np
from choicedesign.criteria import _imat_mnl, _derr

# Swapping algorithm function
def _swapalg(
    design: np.ndarray,init_perf: float,
    n_atts: int, n_alts: int, ncs: int,
    levs: list, cods: list, pars: list, optout: bool, asc: dict, cond: list,
    iter_lim: float, noimprov_lim: float,time_lim: float):
    """Random swapping algorithm

    It optimises an experimental design using a variation of the random swapping 
    algorithm [1].

    References
    ----------
    [1] Quan, W., Rose, J. M., Collins, A. T., & Bliemer, M. C. (2011). A comparison 
    of algorithms for generating efficient choice experiments.
    """
    # Lock design matrix
    desmat = design.copy()

    # Start stopwatch
    t0 = time.time()
    t1 = time.time()

    difftime = 0

    # Initialize algorithm parameters
    i = np.random.choice(np.arange(design.shape[1]))
    t = 0
    ni = 0
    iterperf = init_perf
    newperf = init_perf

    # Start algorithm
    while True:
        
        # Iteration No.
        t = t+1
        
        # If one stopping criterion is satisfied, break!
        if ni >= noimprov_lim or t >= iter_lim or (difftime)/60 >= time_lim:
            break
        
        # Take a random swap
        pairswap = np.random.choice(desmat.shape[0],2,replace=False)
        
        # Check if attribute levels differ
        check_difflevels = desmat[pairswap[0],i] != desmat[pairswap[1],i]

        # If attribute levels differ, do the swap and check for conditions (if defined)
        if check_difflevels:
            swapdes = desmat.copy()
            swapdes[pairswap[0],i] = desmat[pairswap[1],i]
            swapdes[pairswap[1],i] = desmat[pairswap[0],i]
        
            # Check if conditions are satisfied after a swap
            check_all = [True]
            
            # If conditions are defined, this section will check that are satisfied, and rewrite 'check_satisfied_conds' if neccesary
            if cond is not None:
                for c in cond:
                    check_all.append(np.all(eval(c)))

                check_all = np.all(check_all)
            
            # If all conditions are satisfied, compute D-error
            if check_all:
                newperf = _derr(_imat_mnl,swapdes,levs,cods,pars,n_atts,n_alts,ncs,optout,asc)

        # ...else if they do not differ, keep the D-error
        else:
            newperf = iterperf.copy()
            
        # If the swap made an improvement, keep the design and update progress bar
        improved = newperf < iterperf

        if improved:
            desmat = swapdes.copy()
            iterperf = newperf.copy()
            ni = 0
            
            # Update progress bar
            print('Optimizing / ' + 'Elapsed: ' + str(datetime.timedelta(seconds=difftime))[:7] + ' / D-error: ' + str(round(iterperf,6)),end='\r')
        
        # ...else, pass to a random attribute and increment the 'no improvement' counter by 1.
        else:
            i = np.random.choice(np.arange(design.shape[1]))
            ni = ni+1
        
        # Update progress bar each second
        if (difftime)%1 < 0.1:
            print('Optimizing / ' + 'Elapsed: ' + str(datetime.timedelta(seconds=difftime))[:7] + ' / D-error: ' + str(round(iterperf,6)),end='\r',flush=True)
        
        t1 = time.time()
        difftime = t1-t0
    
    # Return optimal design plus efficiency
    return desmat, iterperf, t, difftime