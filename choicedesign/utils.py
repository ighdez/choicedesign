"""Utilitary functions"""

# Import modules
import numpy as np
# from scipy.stats import chi2_contingency

# Function for dummy generation
def _dummygen(x,levs):
    """Generate dummy variables"""
    n_levs = len(levs)

    converted_array = np.empty((x.shape[0],n_levs-1))
    for l in range(n_levs-1):
        converted_array[:,l] = (x == levs[l+1]).astype(int)

    return converted_array

# # Crosstab function
# def _cross(x,y):
#     """Create a cross tab"""
#     tab = []
    
#     for i in np.unique(x):
#         cols = []
        
#         for j in np.unique(y):
#             c = np.count_nonzero((x==i) & (y==j))
#             cols = cols + [c]
        
#         tab = tab + [cols]

#     return np.array(tab)

# Block generation function
def _blockgen(design: np.ndarray, n_blocks: int, ncs: int, reps: int):
    """Blocks generator"""
    # Create array of blocks
    blocks = np.repeat(np.arange(n_blocks)+1,int(ncs/n_blocks))
    # bestcorr = np.inf
    # bestblock = blocks.copy()
    
    for _ in range(reps):
        np.random.shuffle(blocks)
        # blockmat = np.repeat(blocks,int(np.max(design[:,1]))).copy()
        # sumcorr = 0
        
        # for a in range(2,design.shape[1]):
        #     d = design[:,a].copy()
        #     c = _cross(blockmat,d)
        #     corr = chi2_contingency(c)[1]
        #     sumcorr = sumcorr + corr
        
        # if sumcorr < bestcorr:
        #     bestblock = blockmat.copy()
        #     bestcorr = sumcorr

    # return bestblock
    return blocks

# Condition generation function
def _condgen(desname: str, cond: list, names: list, init: bool = False):
    """Conditions generator for the design modules"""    
    # Match variable names with columns in the design matrix
    if init:
        design_columns = [desname + '[i,' + str(i) + ']' for i in range(len(names))]
    else:
        design_columns = [desname + '[:,' + str(i) + ']' for i in range(len(names))]

    # Create new list of conditions
    conditions = []

    for c in cond:
        # Take a copy of the string
        cc = c[:]

        # Replace names by design matrix columns
        for i in range(len(names)):
            cc = cc.replace(names[i],design_columns[i])
        
        # If there is a conditional 'if' statement, then convert it to a logical 'or'
        if 'if' in cc:
            # Split the sting in the 'then' part
            if_part, then_part = cc.split('then')

            # In case there are '&' in the 'if' 'then' part, convert them to '*'
            if_part = if_part.replace('&','*')
            then_part = then_part.replace('&','*')

            # Set the 'logical_not' operator in the 'if' part
            if_part = 'np.logical_not(' + if_part.replace('if','') + ')'

            # Merge if and then parts
            cc = 'np.logical_or(' + if_part + ',' + then_part + ')'
        
        # Finally, append condition to condition list
        conditions.append(cc)
    
    # ...and return the new conditions list
    return conditions

# Generate initial design matrix
def _initdesign(levs: list, ncs: int, cond: list):
    """Generate initial design matrix"""
    # Create and populate the initial design matrix
    desmat = []

    # for k in levs:
    for k in range(len(levs)):
        col = np.array((levs[k] * int(np.ceil(ncs/len(levs[k]))))[:ncs])
        np.random.shuffle(col)
        desmat.append(col)
    
    desmat = np.array(desmat).T

    # Apply conditions if needed
    if cond is not None:
        for i in range(ncs):
            # Check if all conditions are satisfied. If not, do a big enough loop till all conditions are satisfied
            check_all = []

            for c in cond:
                check_all.append(eval(c))

            check_all = np.all(check_all)

            if not check_all:
                for _ in range(10000):
                    # Create a random vector of levels for the row in question
                    for k in range(len(levs)):
                        desmat[i,k] = np.random.choice(levs[k])
                    
                    # Check if conditions are met with the new vector in the design.
                    check_all = []

                    for c in cond:
                        check_all.append(eval(c))

                    check_all = np.all(check_all)


                    # If so, break the loop and go for the next row
                    if check_all:
                        break
            
            # If after the big loop conditions are not met, then raise an error
            assert check_all, 'It is not possible to met all conditions in the initial design matrix.'
    
    # Return the design matrix
    return desmat