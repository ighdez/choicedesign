"""Modules to construct efficient designs"""

# Import modules
from re import L
from unicodedata import name
import pandas as pd
import numpy as np
import datetime

from choicedesign.algorithms import _swapalg
from choicedesign.criteria import _derr, _imat_mnl, _utility_balance_mnl
from choicedesign.utils import _blockgen, _condgen, _initdesign

# RUM-based efficient design
class RUMDesign:
    """RUM-consistent discrete choice experiment design class

    This class allows to create an efficient design for a discrete choice 
    experiment based in a Random Utility Maximisation (RUM) model [1].

    Parameters
    ----------
    atts_list : list[dict]
        List of attributes of the design. Each element is a dictionary that 
        contains the following keys:
            
            - `name`: the attribute name
            - `levels`: a list of attribute levels
            - `coding`: the coding of the attribute levels. Supporte types are 
              'numeric' and 'dummy'.
            - `pars`: prior parameters of the attribute. If `coding = 'numeric' 
              then only one parameter is required. If `coding = 'dummy', then 
              `levels-1` parameters must be used. The first attribute level is 
              taken as a baseline. 
    n_alts : int
        Number of alternatives, apart from the opt-out (if defined).
    self.ncs : int
        Number of choice situations.
    optout : bool, optional
        Should an opt-out be included? If so, it is included as the last alternative, 
        by default False.
    asc : dict, optional
        Dictionary that defines alternative-specific constants (ASC) and their respective 
        prior parameters. Each key is the alternative in which an ASC is desired, and 
        each value is the corresponding prior parameter.
    """

    # Init method
    def __init__(self,atts_list: list, n_alts: int, ncs: int, optout: bool = False, asc: dict = None):

        # Define scalars
        self.N = ncs
        self.J = n_alts
        self.K = len(atts_list)
        self.optout = optout
        self.asc = asc

        # Set names, levels, coding and pars
        self.names = []
        self.levs = []
        self.cods = []
        self.fixed = []
        self.pars = []

        # Start looping among alternatives and attributes
        for j in range(self.J+optout):
            # If ASC are defined, then add them to the lists
            if asc is not None:
                for a in asc:
                    self.names.append('alt' + str(j+1) + '_asc' + str(a) if j < self.J else 'optout_asc' + str(a))
                    self.levs.append([1] if a == (j+1) else [0])
                    self.cods.append('asc')
                    self.fixed.append(1)
                    self.pars.append(asc[j+1] if a == (j+1) else 0)

            # Loop among attributes
            for k in atts_list:
                if j < self.J:
                    self.names.append('alt' + str(j+1) + '_' + k['name'])
                    self.levs.append(k['levels'])
                    self.cods.append(k['coding'])
                    self.fixed.append(0)
                self.pars += k['par']
        
    # Optimise
    def optimise(self, cond: list = None, n_blocks: int = None, iter_lim: int = None, noimprov_lim: int = None, time_lim: int = None, seed: int = None, verbose: bool = False):
        """Create D-efficient RUM design

        Starts the optimisation of the design using a random swapping 
        algorithm and the attributes and prior parameters specified in 
        the `RUMDesign` object. Allows for conditions, blocking and 
        user-defined stopping criteria.

        Parameters
        ----------
        cond : list[str], optional
            List of conditions that the final design must hold. Each element 
            is a string that contains a single condition. Conditions 
            can be of the form of binary relations (e.g., `X > Y` where `X` 
            and `Y` are attributes of a specific alternative) or conditional 
            relations (e.g., `if X > a then Y < b` where `a` and `b` are values).
            Users can specify multiple conditions when the operator `if` is defined, 
            separated by the operator `&`, by default None
        n_blocks : int, optional
            Number of blocks of the final design. Must be a multiple of the number of 
            choice situations, by default None
        iter_lim : int, optional
            Number of iterations before the algorithm stops, by default None
        noimprov_lim : int, optional
            Number of iterations without improvement before the algorithm 
            stops, by default None
        time_lim : int, optional
            Time (in minutes) before the algorithm stops, by default None
        seed : int, optional
            Random seed, by default None
        verbose : bool, optional
            Whether status messages and progress are shown, by default False

        Returns
        -------
        optimal_design : pandas.DataFrame
            The final (optimal) design
        init_perf : float
            D-error of the initial design
        final_perf : float
            D-error of the final design 
        final_iter : int
            Total number of iterations
        ubalance_ratio : float
            Utility balance ratio
        """
        # Generate conditions if defined
        if cond is not None:
            initconds = _condgen('desmat',cond,self.names,init=True)
            algconds = _condgen('swapdes',cond,self.names,init=False)
        else:
            initconds = None
            algconds = None

        # Set random seed if defined
        if seed is not None:
            np.random.seed(seed)

        # Set stopping criteria if defined
        if iter_lim is None:
            iter_lim = np.inf
        
        if noimprov_lim is None:
            noimprov_lim = np.inf
            
        if time_lim is None:
            time_lim = np.inf

        ############################################################
        ########## Step 1: Generate initial design matrix ##########
        ############################################################

        # Generate 10 random designs
        # Keep the one with best performance as initial design
        if verbose:
            print('Generating the initial design matrix')

        desmat = []
        init_perf = np.inf

        for _ in range(10):
            desmat0 = _initdesign(names=self.names,levs=self.levs,ncs=self.N,cond=initconds)
            perf0 = _derr(_imat_mnl,desmat0,self.levs,self.cods,self.pars,self.K,self.J,self.N,self.optout,self.asc)
            if perf0 < init_perf:
                desmat = desmat0.copy()
                init_perf = perf0

        ############################################################
        ############## Step 2: Initialize algorighm ################
        ############################################################

        # Execute Swapping algorithm
        optimal_design, final_perf, final_iter, elapsed_time = _swapalg(
            desmat,init_perf,self.K,self.J,self.N,self.levs,self.cods,self.pars,self.optout,self.asc,algconds,iter_lim,noimprov_lim,time_lim)

        # Compute utility balance ratio
        ubalance_ratio = _utility_balance_mnl(desmat0,self.levs,self.cods,self.pars,self.K,self.J,self.N,self.optout,self.asc)

        ############################################################
        ############## Step 3: Arange final design #################
        ############################################################

        # Add CS column
        optimal_design = np.c_[np.arange(self.N)+1,optimal_design]

        # Generate blocks
        if n_blocks is not None:
            if verbose:
                print('\nGenerating ' + str(n_blocks) + ' blocks...')
            blocksrow = _blockgen(optimal_design,n_blocks,self.N,1000)
            optimal_design = np.c_[optimal_design,blocksrow]

        # Create Pandas DataFrame
        if n_blocks is not None:
            optimal_design = pd.DataFrame(optimal_design,columns=['CS'] + self.names + ['Block'])
        else:
            optimal_design = pd.DataFrame(optimal_design,columns=['CS'] + self.names)

        # Return a summary if verbose is True
        if verbose:
            print('Optimization complete')
            print('Elapsed time: ' + str(datetime.timedelta(seconds=elapsed_time))[:7])
            print('D-error of initial design: ',round(init_perf,6))
            print('D-error of last stored design: ',round(final_perf,6))
            print('Utility Balance ratio: ',round(ubalance_ratio,2),'%')
            print('Algorithm iterations: ',final_iter)
            print('')
        
        # Return the optimal design
        return optimal_design, init_perf, final_perf, final_iter, ubalance_ratio
