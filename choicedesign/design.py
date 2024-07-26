"""Modules to construct experimental designs"""

# Import modules
import pandas as pd
import numpy as np
import datetime
from typing import List
from biogeme.expressions import MonteCarlo, log
from biogeme.models import loglogit, logit
from pyDOE2 import fullfact

from choicedesign.expressions import Attribute
from choicedesign.algorithms import _swapalg
from choicedesign.criteria import _derr, _utility_balance
from choicedesign.utils import _blockgen, _condgen, _initdesign

# Efficient design
class EffDesign:
    """Class for efficient designs for discrete choice experiments

    This class allows to create an efficient design for a discrete choice 
    experiment [1].

    Parameters
    ----------
    X : List[Attribute]
        List of `Attribute` elements.
    ncs : int
        Number of choice situations.
    """

    # Init method
    def __init__(self, X: dict, ncs: int):

        # Define scalars
        self.N = ncs
        self.J = len(X)
 
        # Set names and levels
        self.names = [j.name for j in X]
        self.levs = [j.levels for j in X]

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
    def optimise(self, init_design: pd.DataFrame, V: dict, model: str = 'mnl', draws: int = 1000, iter_lim: int = None, noimprov_lim: int = None, time_lim: int = None, seed: int = None, verbose: bool = False):
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
        # if n_blocks is not None:
        #     if verbose:
        #         print('\nGenerating ' + str(n_blocks) + ' blocks...')
        #     blocksrow = _blockgen(optimal_design,n_blocks,self.N,1000)
        #     optimal_design = np.c_[optimal_design,blocksrow]

        # Create Pandas DataFrame
        # if n_blocks is not None:
        #     optimal_design = pd.DataFrame(optimal_design,columns=['CS'] + self.names + ['Block'])
        # else:
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

    # Generate blocks
    def gen_blocks(self, design: pd.DataFrame, n_blocks: int, n_iter: int = 1000):
        """Generate blocks

        Generate blocks for a design, based on minimising the correlation between the block column and all attributes

        Parameters
        ----------
        design : pd.DataFrame
            Design matrix
        n_blocks : int
            Number of blocks
        n_blocks : int
            Number of iterations of the minsum algorithm, default 1000

        Returns
        -------
        design : pd.DataFrame
            Optimal design with the block column
        """
        blocksrow = _blockgen(design,n_blocks,n_iter)
        design['Block'] = blocksrow

        return design

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
    
# Class for Full-factorial design
class FullFactDesign:
    # Init method
    def __init__(self, X: dict):
 
        # Set names and levels
        self.names = [j.name for j in X]
        self.levs = [j.levels for j in X]

        self.J = len(X)
        self.K = len(self.levs)

    # Generate design matrix
    def gen_design(self):
        # Generate default full-fact design
        n_levels = [len(j) for j in self.levs]
        init_design = fullfact(n_levels)

        # Create pandas dataframe
        init_design = pd.DataFrame(init_design.astype(int),columns=self.names)

        # Replace values with the actual attribute levels
        for k in range(self.K):
            init_design.iloc[:,k].replace(np.arange(n_levels[k]),self.levs[k],inplace=True)

        return init_design