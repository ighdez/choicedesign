"""Utility functions for design generation and condition handling.

Contains helpers for:

- Generating balanced initial random designs (``_initdesign``).
- Parsing condition strings into callable predicates (``_parse_condition``).
- Searching for block assignments that minimise attribute correlation (``_blockgen``).
- Dummy-coding categorical attribute columns (``_dummygen``).
"""

# Import modules
import re
import numpy as np
import pandas as pd

# Function for dummy generation
def _dummygen(x, levs):
    """Generate dummy variables for a categorical attribute column.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Column of observed attribute values.
    levs : list
        Ordered list of all attribute levels. The first level is the
        reference category and is excluded from the output.

    Returns
    -------
    np.ndarray, shape (n, len(levs) - 1)
        Binary indicator matrix (1 where ``x == level``, 0 otherwise).
    """
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
def _blockgen(design: pd.DataFrame, n_blocks: int, reps: int):
    """Search for a block assignment that minimises attribute correlation.

    Parameters
    ----------
    design : pandas.DataFrame
        Optimised design (must include a ``CS`` column in position 0,
        which is skipped when computing correlations).
    n_blocks : int
        Number of blocks. Must divide evenly into ``len(design)``.
    reps : int
        Number of random permutations to evaluate.

    Returns
    -------
    bestblock : np.ndarray, shape (ncs,)
        Block labels (1-indexed) for each choice situation.
    corr_list : list[float]
        Running best total absolute correlation at each improvement step.
    """

    ncs = len(design)

    # Create array of blocks
    blocks = np.repeat(np.arange(n_blocks)+1,int(ncs/n_blocks))

    bestcorr = np.inf
    bestblock = blocks.copy()
    corr_list = []

    design_array = design.iloc[:,1:]

    for _ in range(reps):
        np.random.shuffle(blocks)
        design_array['cand_block'] = blocks

        sumcorr = design_array.corr()['cand_block'].abs().sum()
        
        if sumcorr < bestcorr:
            bestblock = blocks.copy()
            bestcorr = sumcorr
            corr_list.append(bestcorr)

    # return bestblock
    return bestblock, corr_list

# Condition parsing function
def _parse_condition(cond_str: str, names: list):
    """Parse a condition string into a callable.

    Supports binary conditions ('A > B'), if/then conditionals
    ('if A > v then B < w'), and & compounds ('A > v & B < w').
    Attribute names are resolved to column indices at parse time so
    any typo raises ValueError immediately, before the design loop runs.

    Parameters
    ----------
    cond_str : str
        A single condition string.
    names : list[str]
        Ordered list of attribute names matching the design matrix columns.

    Returns
    -------
    callable
        f(desmat: np.ndarray) -> np.ndarray[bool], where desmat has
        shape (n_rows, n_attributes).
    """

    _OPS = {
        '>=': lambda a, b: a >= b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: a == b,
        '!=': lambda a, b: a != b,
        '>':  lambda a, b: a > b,
        '<':  lambda a, b: a < b,
    }

    def _resolve(token):
        token = token.strip()
        if token in names:
            idx = names.index(token)
            return lambda desmat, i=idx: desmat[:, i]
        try:
            val = float(token)
            return lambda desmat, v=val: np.full(desmat.shape[0], v)
        except ValueError:
            raise ValueError(
                f"'{token}' is not a known attribute or a numeric value. "
                f"Known attributes: {names}"
            )

    def _parse_atomic(s):
        s = s.strip()
        match = re.match(r'^(.+?)\s*(>=|<=|==|!=|>|<)\s*(.+)$', s)
        if not match:
            raise ValueError(f"Cannot parse condition fragment: '{s}'")
        left_token = match.group(1).strip()
        op          = match.group(2)
        right_token = match.group(3).strip()

        if left_token not in names:
            raise ValueError(
                f"Unknown attribute '{left_token}' in condition '{cond_str}'. "
                f"Known attributes: {names}"
            )

        left_fn = _resolve(left_token)
        right_fn = _resolve(right_token)
        op_fn = _OPS[op]

        return lambda desmat, l=left_fn, r=right_fn, o=op_fn: o(l(desmat), r(desmat))

    def _parse_and(s):
        parts = [p.strip() for p in s.split('&')]
        fns = [_parse_atomic(p) for p in parts if p]
        if not fns:
            raise ValueError(f"Empty condition fragment: '{s}'")
        if len(fns) == 1:
            return fns[0]
        def combined(desmat, fns=fns):
            result = fns[0](desmat)
            for fn in fns[1:]:
                result = result & fn(desmat)
            return result
        return combined

    cond_str = cond_str.strip()

    # if/then: 'if <antecedent> then <consequent>'
    # Semantics: NOT(antecedent) OR consequent  (material implication)
    ifthen = re.match(r'^if\s+(.+?)\s+then\s+(.+)$', cond_str, re.IGNORECASE)
    if ifthen:
        if_fn   = _parse_and(ifthen.group(1))
        then_fn = _parse_and(ifthen.group(2))
        return lambda desmat, i=if_fn, t=then_fn: (
            np.logical_or(np.logical_not(i(desmat)), t(desmat))
        )

    return _parse_and(cond_str)


# Generate initial design matrix
def _initdesign(levs: list, ncs: int, cond: list):
    """Generate initial design matrix

    Parameters
    ----------
    levs : list
        List of level arrays, one per attribute.
    ncs : int
        Number of choice situations (rows).
    cond : list[callable] or None
        Parsed condition callables from `_parse_condition`. Each callable
        accepts a 2D array of shape (n_rows, n_attrs) and returns a bool array.
    """
    desmat = []

    for k in range(len(levs)):
        col = np.array((levs[k] * int(np.ceil(ncs/len(levs[k]))))[:ncs])
        np.random.shuffle(col)
        desmat.append(col)

    desmat = np.array(desmat).T

    if cond is not None:
        for i in range(ncs):
            row = desmat[i:i+1, :]
            satisfied = np.all([np.all(c(row)) for c in cond])

            if not satisfied:
                for _ in range(10000):
                    for k in range(len(levs)):
                        desmat[i, k] = np.random.choice(levs[k])

                    row = desmat[i:i+1, :]
                    satisfied = np.all([np.all(c(row)) for c in cond])

                    if satisfied:
                        break

            if not satisfied:
                raise ValueError(
                    f'Could not satisfy all conditions at row {i} after 10000 attempts. '
                    'The conditions may be too restrictive for the given attribute levels.'
                )

    return desmat