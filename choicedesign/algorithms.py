"""Optimisation algorithms"""

# Load modules
import datetime
import itertools
import time
from typing import Callable
import pandas as pd
import numpy as np
from choicedesign.criteria import MNLModel

# Swapping algorithm function
def _swapalg(
    design: pd.DataFrame, model: MNLModel,
    init_perf: float, cond: list,
    iter_lim: float, noimprov_lim: float, time_lim: float,
    derr_fn: Callable):
    """Random swapping algorithm

    It optimises an experimental design using a variation of the random swapping 
    algorithm [1].

    References
    ----------
    [1] Quan, W., Rose, J. M., Collins, A. T., & Bliemer, M. C. (2011). A comparison 
    of algorithms for generating efficient choice experiments.
    """
    # Lock design matrix and names
    names = design.columns
    desmat = design.to_numpy()

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

    # Initialize candidate swaps list
    candidate_swaps = []
    
    for _ in range(desmat.shape[1]):
        candidate_swaps.append(range(len(desmat)))

    # Start algorithm
    while True:
        
        # Iteration No.
        t = t+1
        
        # If one stopping criterion is satisfied, break!
        if ni >= noimprov_lim or t >= iter_lim or (difftime)/60 >= time_lim:
            break
        
        # Take a random swap
        pairswap = np.random.choice(candidate_swaps[i],2,replace=False)
        
        # Check if attribute levels differ
        check_difflevels = desmat[pairswap[0],i] != desmat[pairswap[1],i]

        # If attribute levels differ, do the swap and check for conditions (if defined)
        if check_difflevels:
            swapdes = desmat.copy()
            swapdes[pairswap[0],i] = desmat[pairswap[1],i]
            swapdes[pairswap[1],i] = desmat[pairswap[0],i]
        
            # Check if all conditions are satisfied after the swap
            if cond is not None:
                check_all = np.all([np.all(c(swapdes)) for c in cond])
            else:
                check_all = True
            
            # If all conditions are satisfied, compute D-error
            if check_all:
                newperf = derr_fn(pd.DataFrame(swapdes,columns=names),model)

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
    return pd.DataFrame(desmat,columns=names), iterperf, t, difftime


# RSC algorithm function
def _rscalg(
    design: pd.DataFrame, model: MNLModel,
    init_perf: float, cond: list,
    iter_lim: float, noimprov_lim: float, time_lim: float,
    derr_fn: Callable):
    """Random RSC (Relabelling, Swapping, Cycling) algorithm

    Optimises an experimental design by applying one of three random moves
    per iteration: relabelling (swap all occurrences of two level values in a
    column), swapping (swap two individual row values in a column), or cycling
    (rotate all values in a column by one position). The move and column are
    chosen randomly each iteration; improvements are kept.

    References
    ----------
    [1] Quan, W., Rose, J. M., Collins, A. T., & Bliemer, M. C. (2011). A comparison
    of algorithms for generating efficient choice experiments.
    """
    names = design.columns
    desmat = design.to_numpy()

    t0 = time.time()
    difftime = 0

    t = 0
    ni = 0
    iterperf = init_perf
    newperf = init_perf

    while True:

        t += 1

        if ni >= noimprov_lim or t >= iter_lim or difftime / 60 >= time_lim:
            break

        # Pick a random column and move type
        i = np.random.choice(np.arange(desmat.shape[1]))
        move = np.random.choice(['R', 'S', 'C'])

        swapdes = desmat.copy()
        valid_move = True

        if move == 'R':
            # Relabelling: swap all occurrences of two randomly chosen level values
            col_levels = np.unique(desmat[:, i])
            if len(col_levels) < 2:
                valid_move = False
            else:
                l1, l2 = np.random.choice(col_levels, 2, replace=False)
                swapdes[desmat[:, i] == l1, i] = l2
                swapdes[desmat[:, i] == l2, i] = l1

        elif move == 'S':
            # Swapping: swap two individual row values in the column
            pairswap = np.random.choice(len(desmat), 2, replace=False)
            if desmat[pairswap[0], i] == desmat[pairswap[1], i]:
                valid_move = False
            else:
                swapdes[pairswap[0], i] = desmat[pairswap[1], i]
                swapdes[pairswap[1], i] = desmat[pairswap[0], i]

        elif move == 'C':
            # Cycling: rotate all values in the column by one position
            swapdes[:, i] = np.roll(desmat[:, i], 1)
            if np.array_equal(swapdes[:, i], desmat[:, i]):
                valid_move = False

        if valid_move:
            if cond is not None:
                check_all = np.all([np.all(c(swapdes)) for c in cond])
            else:
                check_all = True

            if check_all:
                newperf = derr_fn(pd.DataFrame(swapdes, columns=names), model)
            else:
                newperf = iterperf
        else:
            newperf = iterperf

        improved = newperf < iterperf

        if improved:
            desmat = swapdes.copy()
            iterperf = newperf.copy()
            ni = 0
            print('Optimizing / ' + 'Elapsed: ' + str(datetime.timedelta(seconds=difftime))[:7] + ' / D-error: ' + str(round(iterperf, 6)), end='\r')
        else:
            ni += 1

        if difftime % 1 < 0.1:
            print('Optimizing / ' + 'Elapsed: ' + str(datetime.timedelta(seconds=difftime))[:7] + ' / D-error: ' + str(round(iterperf, 6)), end='\r', flush=True)

        t1 = time.time()
        difftime = t1 - t0

    return pd.DataFrame(desmat, columns=names), iterperf, t, difftime


# Modified Federov algorithm function
def _federovalg(
    design: pd.DataFrame, model: MNLModel,
    init_perf: float, cond: list,
    iter_lim: float, noimprov_lim: float, time_lim: float,
    derr_fn: Callable, levs: list):
    """Modified Federov algorithm

    Optimises an experimental design by replacing one row at a time with the
    best available candidate from the full factorial of attribute levels. Each
    iteration picks a random row and evaluates all candidates; the replacement
    that yields the greatest D-error improvement is kept.

    The candidate set is built from the full factorial and pre-filtered by
    conditions once at startup (lazy filtering: each candidate row is checked
    independently).

    References
    ----------
    [1] Quan, W., Rose, J. M., Collins, A. T., & Bliemer, M. C. (2011). A comparison
    of algorithms for generating efficient choice experiments.
    """
    names = design.columns
    desmat = design.to_numpy()

    # Build candidate set from full factorial, then pre-filter by conditions
    candidate_set = np.array(list(itertools.product(*levs)))
    if cond is not None:
        mask = np.array([
            np.all([np.all(c(row.reshape(1, -1))) for c in cond])
            for row in candidate_set
        ])
        candidate_set = candidate_set[mask]

    t0 = time.time()
    difftime = 0

    t = 0
    ni = 0
    iterperf = init_perf

    while True:

        t += 1

        if ni >= noimprov_lim or t >= iter_lim or difftime / 60 >= time_lim:
            break

        # Pick a random row to replace
        r = np.random.choice(len(desmat))

        # Try every candidate as a replacement for row r, keep the best
        best_perf = iterperf
        best_candidate = None

        for candidate in candidate_set:
            if np.array_equal(candidate, desmat[r]):
                continue
            swapdes = desmat.copy()
            swapdes[r] = candidate
            newperf = derr_fn(pd.DataFrame(swapdes, columns=names), model)
            if newperf < best_perf:
                best_perf = newperf
                best_candidate = candidate

        if best_candidate is not None:
            desmat[r] = best_candidate
            iterperf = best_perf
            ni = 0
            print('Optimizing / ' + 'Elapsed: ' + str(datetime.timedelta(seconds=difftime))[:7] + ' / D-error: ' + str(round(iterperf, 6)), end='\r')
        else:
            ni += 1

        if difftime % 1 < 0.1:
            print('Optimizing / ' + 'Elapsed: ' + str(datetime.timedelta(seconds=difftime))[:7] + ' / D-error: ' + str(round(iterperf, 6)), end='\r', flush=True)

        t1 = time.time()
        difftime = t1 - t0

    return pd.DataFrame(desmat, columns=names), iterperf, t, difftime