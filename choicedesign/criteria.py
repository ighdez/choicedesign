"""Optimality criteria for discrete choice experimental designs.

Provides:

- :class:`MNLModel` — a pre-compiled MNL model used inside the optimisation loop.
- ``_derr`` — D-error (D-efficient design criterion).
- ``_db_derr`` — Db-error (Bayesian D-efficient, averaged over prior draws).
- ``_aerr`` — A-error (average parameter variance).
- ``_cerr`` — C-error (WTP variance via delta method).
- ``_utility_balance`` — utility balance ratio.
"""

import numpy as np
import pandas as pd
from typing import List

from choicedesign.expressions import Expression, Parameter, ASC, get_unique_params


class MNLModel:
    """Pre-compiled MNL model for efficient repeated evaluation.

    Symbolic differentiation of the utility functions with respect to
    each parameter is performed **once** at construction time.  All
    subsequent calls (information matrix, probabilities, D-error) are
    pure NumPy operations, making this suitable for the inner loop of
    the swapping algorithm.

    Parameters
    ----------
    V : dict[int, Expression]
        Utility functions, e.g. ``{1: V1, 2: V2}``.
    params : list[Parameter], optional
        Ordered list of parameters.  If *None*, extracted automatically
        from ``V`` via :func:`get_unique_params`.
    """

    def __init__(self, V: dict, params: List[Parameter] = None):
        self.V = V
        self.params = params if params is not None else get_unique_params(V)

        # Pre-compute symbolic derivative expressions: deriv_exprs[j][k]
        # so we never pay the Python-tree traversal cost inside the hot loop.
        self._deriv_exprs: List[List[Expression]] = []
        for v_expr in V.values():
            self._deriv_exprs.append(
                [v_expr.differentiate(p) for p in self.params]
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_utilities(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate utility functions.

        Returns
        -------
        np.ndarray, shape (ncs, J)
        """
        ncs = len(data)
        J = len(self.V)
        U = np.empty((ncs, J))
        for j, v_expr in enumerate(self.V.values()):
            val = v_expr.evaluate(data)
            U[:, j] = val if not np.isscalar(val) else val
        return U

    def _eval_probs(self, data: pd.DataFrame) -> np.ndarray:
        """Compute MNL choice probabilities.

        Returns
        -------
        np.ndarray, shape (ncs, J)
        """
        U = self._eval_utilities(data)
        # Subtract row-wise max for numerical stability before exp
        U -= U.max(axis=1, keepdims=True)
        eU = np.exp(U)
        return eU / eU.sum(axis=1, keepdims=True)

    def _eval_gradient_tensor(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate the gradient tensor ∂V_j/∂β_k over all rows.

        Returns
        -------
        np.ndarray, shape (ncs, J, K)
            Entry [n, j, k] = ∂V_j/∂β_k evaluated at design row n.
        """
        ncs = len(data)
        J = len(self.V)
        K = len(self.params)
        G = np.zeros((ncs, J, K))
        for j, deriv_row in enumerate(self._deriv_exprs):
            for k, deriv_expr in enumerate(deriv_row):
                val = deriv_expr.evaluate(data)
                G[:, j, k] = val if not np.isscalar(val) else val
        return G

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def information_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Compute the MNL Fisher information matrix.

        The information matrix is:

        .. math::

            I_{kl} = \\sum_n \\left[
                \\sum_j P_{nj}\\, x_{njk}\\, x_{njl}
                - \\left(\\sum_j P_{nj}\\, x_{njk}\\right)
                  \\left(\\sum_j P_{nj}\\, x_{njl}\\right)
            \\right]

        where :math:`x_{njk} = \\partial V_j / \\partial \\beta_k`
        at design row *n*.

        Parameters
        ----------
        data : pd.DataFrame
            Design matrix (rows = choice situations).

        Returns
        -------
        np.ndarray, shape (K, K)
        """
        P = self._eval_probs(data)           # (ncs, J)
        G = self._eval_gradient_tensor(data)  # (ncs, J, K)

        # term1[k,l] = Σ_n Σ_j P_nj · x_njk · x_njl
        term1 = np.einsum('nj,njk,njl->kl', P, G, G)

        # weighted_g[n,k] = Σ_j P_nj · x_njk
        weighted_g = np.einsum('nj,njk->nk', P, G)  # (ncs, K)

        # term2[k,l] = Σ_n (Σ_j P_nj x_njk)(Σ_j P_nj x_njl)
        term2 = np.einsum('nk,nl->kl', weighted_g, weighted_g)

        return term1 - term2

    def probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """Return MNL choice probabilities.

        Returns
        -------
        np.ndarray, shape (ncs, J)
        """
        return self._eval_probs(data)


# ---------------------------------------------------------------------------
# Criteria functions
# ---------------------------------------------------------------------------

def _derr(design: pd.DataFrame, model: MNLModel) -> float:
    """D-error of a design given a compiled MNL model.

    The D-error is defined as:

    .. math::

        D\\text{-error} = \\det\\left(I(\\beta)^{-1}\\right)^{1/K}

    where *K* is the number of **non-ASC** parameters.  When ASCs are
    present in the model their rows and columns are excluded from the
    information matrix before inversion, because ASCs are not part of
    the experimental design being optimised.  Returns ``np.inf`` when
    the (reduced) information matrix is singular or its inverse has a
    non-positive determinant.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    model : MNLModel
        Pre-compiled MNL model.

    Returns
    -------
    float
    """
    im = model.information_matrix(design)

    # Drop rows/columns that correspond to ASC parameters
    non_asc_idx = [i for i, p in enumerate(model.params) if not isinstance(p, ASC)]
    if len(non_asc_idx) < im.shape[0]:
        ix = np.ix_(non_asc_idx, non_asc_idx)
        im = im[ix]

    if np.isclose(np.linalg.det(im), 0):
        return np.inf

    vce = np.linalg.solve(im, np.eye(im.shape[0]))
    detvce = np.linalg.det(vce)

    if detvce > 0:
        return detvce ** (1 / vce.shape[0])

    return np.inf


def _db_derr(
    design: pd.DataFrame,
    model: MNLModel,
    n_draws: int,
    rng: np.random.Generator,
) -> float:
    """Db-error of a design: expected D-error over uncertain prior distributions.

    For each parameter with a ``prior_std`` set, draws are taken from
    ``Normal(prior, prior_std)``.  The D-error is evaluated for every draw
    and the mean is returned as the Db-error.  Parameters without
    ``prior_std`` (or with ``prior_std=None``) are held fixed.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    model : MNLModel
        Pre-compiled MNL model.
    n_draws : int
        Number of Monte Carlo draws from the prior distributions.
    rng : np.random.Generator
        NumPy random generator. Pass a single instance from the caller so
        that the full optimisation run uses one coherent random stream.

    Returns
    -------
    float
    """
    uncertain = [p for p in model.params if getattr(p, 'prior_std', None)]

    if not uncertain:
        return _derr(design, model)

    # Store the user-specified prior means before any mutation
    prior_means = {p: p.prior for p in uncertain}

    total = 0.0
    for _ in range(n_draws):
        for p in uncertain:
            p.prior = rng.normal(prior_means[p], p.prior_std)
        total += _derr(design, model)

    # Restore original prior means
    for p in uncertain:
        p.prior = prior_means[p]

    return total / n_draws


def _aerr(design: pd.DataFrame, model: MNLModel) -> float:
    """A-error of a design: average variance across all K non-ASC parameters.

    .. math::

        A\\text{-error} = \\operatorname{trace}\\left(I^{-1}\\right) / K

    Returns ``np.inf`` when the information matrix is singular.
    """
    im = model.information_matrix(design)

    non_asc_idx = [i for i, p in enumerate(model.params) if not isinstance(p, ASC)]
    if len(non_asc_idx) < im.shape[0]:
        ix = np.ix_(non_asc_idx, non_asc_idx)
        im = im[ix]

    if np.isclose(np.linalg.det(im), 0):
        return np.inf

    vce = np.linalg.solve(im, np.eye(im.shape[0]))
    return np.trace(vce) / im.shape[0]


def _cerr(
    design: pd.DataFrame,
    model: MNLModel,
    cost_param: Parameter,
    wtp_params: List[Parameter],
) -> float:
    """C-error of a design: sum of WTP variances (delta method).

    For each ``wtp_param`` β_x the WTP = β_x / β_cost has variance

    .. math::

        c_i^\\top I^{-1} c_i, \\quad
        c_i[x] = 1/\\beta_{\\text{cost}},\\;
        c_i[\\text{cost}] = -\\beta_x / \\beta_{\\text{cost}}^2

    The C-error is the sum over all nominated WTPs.  Returns ``np.inf``
    when the information matrix is singular.

    Parameters
    ----------
    cost_param : Parameter
        The cost (denominator) parameter.
    wtp_params : list[Parameter]
        Parameters whose WTP ratios are to be optimised.
    """
    im = model.information_matrix(design)

    non_asc_params = [p for p in model.params if not isinstance(p, ASC)]
    non_asc_idx = [i for i, p in enumerate(model.params) if not isinstance(p, ASC)]
    if len(non_asc_idx) < im.shape[0]:
        ix = np.ix_(non_asc_idx, non_asc_idx)
        im = im[ix]

    if np.isclose(np.linalg.det(im), 0):
        return np.inf

    vce = np.linalg.solve(im, np.eye(im.shape[0]))

    param_names = [p.name for p in non_asc_params]
    cost_idx = param_names.index(cost_param.name)
    K = len(non_asc_params)

    total = 0.0
    for wtp_p in wtp_params:
        wtp_idx = param_names.index(wtp_p.name)
        c = np.zeros(K)
        c[wtp_idx] = 1.0 / cost_param.prior
        c[cost_idx] = -wtp_p.prior / (cost_param.prior ** 2)
        total += c @ vce @ c

    return total


def _utility_balance(design: pd.DataFrame, model: MNLModel) -> float:
    """Utility balance ratio of a design.

    A value of 100 % indicates perfectly equal market shares across all
    alternatives (maximum utility balance).  A value near 0 % indicates
    near-strict dominance of one alternative.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix.
    model : MNLModel
        Pre-compiled MNL model.

    Returns
    -------
    float
        Utility balance ratio in [0, 100].
    """
    P = model.probabilities(design)   # (ncs, J)
    J = P.shape[1]
    ncs = P.shape[0]

    B = P / (1.0 / J)                 # (ncs, J)
    B = np.prod(B, axis=1) * 100      # (ncs,)
    return float(np.sum(B) / ncs)
