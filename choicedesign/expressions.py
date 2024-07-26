"""ChoiceDesign expressions"""

# Import modules
from typing import Union
from biogeme.expressions import Variable, Beta

# Class of attributes
class Attribute(Variable):
    """Class of attributes

    This is the class that defines attributes for the experimental design

    Parameters
    ----------
    name : str
        The attribute name, as it will appear in the design matrix
    levels : Union[int,list]
        Possible attribute levels. If `int`, the attribute levels will 
        be a consecutive list of integers from 0 to `levels`. If `list`, 
        each element is a specific attribute level.
    """
    
    # Init method
    def __init__(self, name: str, levels: Union[int,list]):
        super().__init__(name)
        self.__class__ = Variable

        # Index of the variable
        self.variableId = None
        # self.name = name

        # # Generate set of attribute levels
        # if isinstance(levels,int):
        #     if levels < 2:
        #         raise ValueError('levels must be greater or equal than two')
        # elif isinstance(levels,list):
        #     if len(levels)<2:
        #         raise ValueError('levels must be greater or equal than two')

        self.levels = levels

# Class of attribute parameters
class Parameter(Beta):
    # Init method
    def __init__(self, name: str, prior: float):
        """Class of attribute parameters

        This is the class that defines attribute parameters for the experimental design

        Parameters
        ----------
        name : str
            The parameter name
        prior : float
            Prior value
        """
        super().__init__(name,prior,None,None,0)

        self.__class__ = Beta

# Class of ASCs
class ASC(Beta):
    # Init method
    def __init__(self, name: str, prior: float):
        """Class of alternative-specific constants

        This is the class that defines alternative-specific constants (ASC) for the experimental design

        Parameters
        ----------
        name : str
            The parameter name
        prior : float
            Prior value
        """
        super().__init__(name,prior,None,None,1)

        self.__class__ = Beta