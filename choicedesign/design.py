"""Modules to construct efficient designs"""

# Import modules
from re import L
from unicodedata import name
import pandas as pd
import numpy as np
import datetime
from biogeme.expressions import Expression, MonteCarlo, log
from biogeme.models import loglogit, logit

from choicedesign.algorithms import _swapalg
from choicedesign.criteria import _derr, _utility_balance
from choicedesign.utils import _blockgen, _condgen, _initdesign

# Efficient design
class EffDesign:
    """RUM-consistent discrete choice experiment design class

    This class allows to create an efficient design for a discrete choice 
    experiment [1].

    Parameters
    ----------
    alts : list
        List of alternative names, to generate the design matrix
    ncs : int
        Number of choice situations.
    atts_list : list[dict]
        List of attributes of the design. Each element is a dictionary that 
        contains the following keys:
            
            - `name`: the attribute name
            - `levels`: a list of attribute levels
            - `avail`: a list that details whether the attribute is part of 
            a specific alternative. Each element is one of the alternative 
            names of the parameter `alts`.
    """

    # Init method
    def __init__(self,atts_list: list, alts: list, ncs: int):

        # Define scalars
        self.N = ncs
        self.J = len(alts)
        self.K = len(atts_list)
 
        # Set names and levels
        self.names = []
        self.levs = []

        for j in alts:
            for k in atts_list:
                if j in k['avail']:
                    self.names.append(j + '_' + k['name'])
                    self.levs.append(k['levels'])

    # Generate initial design matrix
    def gen_initdesign(self,cond: list = None, seed: bool = None):
        """Generate initial design matrix

        It generates the initial design matrix. The user can define a set of
        conditions that must be satisfied.

        Parameters
        -------
        cond : list[str], optional
            List of conditions that the final design must hold. Each element 
            is a string that contains a single condition. Conditions 
            can be of the form of binary relations (e.g., `X > Y` where `X` 
            and `Y` are attributes of a specific alternative) or conditional 
            relations (e.g., `if X > a then Y < b` where `a` and `b` are values).
            Users can specify multiple conditions when the operator `if` is defined, 
            separated by the operator `&`, by default None
        seed : bool, None
            Random seed, by default None

        Returns
        -------
        init_design : pandas.DataFrame
            A Pandas DataFrame with the initial design matrix.
        """

        # Generate conditions if defined
        if cond is not None:
            self.initconds = _condgen('desmat',cond,self.names,init=True)
            self.algconds = _condgen('swapdes',cond,self.names,init=False)
        else:
            self.initconds = None
            self.algconds = None

        # Set random seed if defined
        if seed is not None:
            np.random.seed(seed)

        # Generate initial design matrix
        init_design = _initdesign(levs=self.levs,ncs=self.N,cond=self.initconds)

        return pd.DataFrame(init_design,columns=self.names)

    # Optimise
    def optimise(self, init_design: pd.DataFrame, V: dict, model: str = 'mnl', draws: int = 1000, n_blocks: int = None, iter_lim: int = None, noimprov_lim: int = None, time_lim: int = None, seed: int = None, verbose: bool = False):
        """Create D-efficient RUM design

        Starts the optimisation of the design using a random swapping 
        algorithm and the attributes and prior parameters specified in 
        the `RUMDesign` object. Allows for conditions, blocking and 
        user-defined stopping criteria.

        Parameters
        ----------
        initial_design : pandas.DataFrame
            The initial design matrix as a Pandas DataFrame
        V : dict
            A dictionary with the utility function, using the same syntax as in Biogeme
        model : str
            The base model for the efficient design, by default 'mnl'
        draws : int, optional
            Number of draws for the Monte Carlo integration, by default 1000
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
        ########## Step 1: Set initial design performance ##########
        ############################################################

        if verbose:
            print('Evaluating initial design')

        desmat = init_design

        models_ubalance = []
        av = dict()
        if model == 'mnl':
            for k, _ in V.items():
                av[k] = 1
            model_object = loglogit(V,av,1)

            for k, _ in V.items():
                models_ubalance.append(logit(V,av,k))
        elif model == 'mnl_bayesian':
            for k, _ in V.items():
                av[k] = 1
            model_object = log(MonteCarlo(logit(V,av,1)))

            for k, _ in V.items():
                models_ubalance.append(MonteCarlo(logit(V,av,k)))
        else:
            raise ValueError("""Model name must be either 'mnl' or 'mnl_bayesian'""")

        init_perf = _derr(desmat,model_object,draws)

        ############################################################
        ############## Step 2: Initialize algorighm ################
        ############################################################

        # Execute Swapping algorithm
        optimal_design, final_perf, final_iter, elapsed_time = _swapalg(
            desmat,model_object,draws,init_perf,self.algconds,iter_lim,noimprov_lim,time_lim)

        # Compute utility balance ratio
        ubalance_ratio = _utility_balance(pd.DataFrame(optimal_design,columns=self.names),models_ubalance,draws)

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

    # Evaluate
    def evaluate(self, design: pd.DataFrame, V: dict, model: str = 'mnl', draws: int = 1000):
        """Evaluate design

        Evaluates a design stored in a Pandas data frame

        Parameters
        ----------
        design : pd.DataFrame
            Design to evaluate
        V : dict
            A dictionary with the utility function, using the same syntax as in Biogeme
        model : str
            The base model for the efficient design, by default 'mnl'

        Returns
        -------
        perf : float
            The D-error of the design
        ubalance_ratio : float
            Utility balance ratio
        """
        # Drop CS column and Block (if present) from pandas dataframe
        desmat = design.drop('CS',axis=1)

        if 'Block' in desmat.columns:
            desmat = desmat.drop('Block',axis=1)

        models_ubalance = []
        av = dict()
        if model == 'mnl':
            for k, _ in V.items():
                av[k] = 1
            model_object = loglogit(V,av,1)

            for k, _ in V.items():
                models_ubalance.append(logit(V,av,k))
        elif model == 'mnl_bayesian':
            for k, _ in V.items():
                av[k] = 1
            model_object = log(MonteCarlo(logit(V,av,1)))

            for k, _ in V.items():
                models_ubalance.append(MonteCarlo(logit(V,av,k)))
        else:
            raise ValueError("""Model name must be either 'mnl' or 'mnl_bayesian'""")
        
        # Evaluate the performance and utility balance of the design
        perf = _derr(desmat,model_object,draws)
        ubalance_ratio = _utility_balance(desmat,models_ubalance,draws)

        # Return performance and utility balance
        return perf, ubalance_ratio