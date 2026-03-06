"""ChoiceDesign native expression system"""

import numpy as np
from typing import Union, List


def _wrap(value):
    """Wrap a Python scalar in a Constant expression if not already an Expression."""
    if isinstance(value, Expression):
        return value
    return Constant(value)


class Expression:
    """Base class for all ChoiceDesign expressions.

    Subclasses implement `evaluate(data, draws)` and optionally
    `differentiate(param)` for symbolic differentiation.
    Arithmetic operators are overloaded so that utility functions
    can be written naturally:

        V1 = beta_A * alt1_A + beta_B * alt1_B
    """

    def evaluate(self, data=None, draws=None):
        """Evaluate the expression and return a NumPy array or scalar.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Design matrix. Required for Attribute expressions.
        draws : int, optional
            Number of Monte Carlo draws. Required for RandomParameter.

        Returns
        -------
        np.ndarray or float
        """
        raise NotImplementedError

    def get_params(self) -> list:
        """Return a list of all Parameter objects in the expression tree.

        The list may contain duplicates; use `get_unique_params` for
        a deduplicated version.
        """
        return []

    def differentiate(self, param: "Parameter") -> "Expression":
        """Return the symbolic derivative with respect to ``param``.

        Parameters
        ----------
        param : Parameter
            The parameter to differentiate with respect to.

        Returns
        -------
        Expression
        """
        return Constant(0)

    # -- Arithmetic operators -------------------------------------------------

    def __add__(self, other):  return Add(self, _wrap(other))
    def __radd__(self, other): return Add(_wrap(other), self)
    def __sub__(self, other):  return Sub(self, _wrap(other))
    def __rsub__(self, other): return Sub(_wrap(other), self)
    def __mul__(self, other):  return Mul(self, _wrap(other))
    def __rmul__(self, other): return Mul(_wrap(other), self)
    def __truediv__(self, other):  return Div(self, _wrap(other))
    def __rtruediv__(self, other): return Div(_wrap(other), self)
    def __pow__(self, other):  return Pow(self, _wrap(other))
    def __rpow__(self, other): return Pow(_wrap(other), self)
    def __neg__(self):         return Neg(self)

    # -- Comparison operators (return indicator expressions, not booleans) ----
    # __hash__ must stay consistent after overriding __eq__.
    __hash__ = object.__hash__

    def __eq__(self, other):  return Equal(self, _wrap(other))
    def __ne__(self, other):  return NotEqual(self, _wrap(other))
    def __lt__(self, other):  return LessThan(self, _wrap(other))
    def __le__(self, other):  return LessEqual(self, _wrap(other))
    def __gt__(self, other):  return GreaterThan(self, _wrap(other))
    def __ge__(self, other):  return GreaterEqual(self, _wrap(other))


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

class Constant(Expression):
    """A fixed numeric constant."""

    def __init__(self, value: float):
        self.value = value

    def evaluate(self, data=None, draws=None):
        return np.float64(self.value)

    def differentiate(self, param):
        return Constant(0)

    def __repr__(self):
        return f"Constant({self.value})"


class Attribute(Expression):
    """A design attribute — a column of the design matrix.

    Parameters
    ----------
    name : str
        Column name as it appears in the design DataFrame.
    levels : list
        Discrete levels this attribute can take.
    """

    def __init__(self, name: str, levels: Union[int, list]):
        self.name = name
        self.levels = levels if isinstance(levels, list) else list(range(levels))

    def evaluate(self, data=None, draws=None):
        if data is None:
            raise ValueError(f"Data required to evaluate Attribute '{self.name}'")
        return data[self.name].values.astype(float)

    def differentiate(self, param):
        return Constant(0)

    def __repr__(self):
        return f"Attribute('{self.name}', levels={self.levels})"


class Parameter(Expression):
    """A model parameter with a fixed prior value.

    Parameters
    ----------
    name : str
        Parameter name.
    prior : float
        Prior (point) value used in D-efficient design.
    """

    def __init__(self, name: str, prior: float):
        self.name = name
        self.prior = prior

    def evaluate(self, data=None, draws=None):
        return np.float64(self.prior)

    def get_params(self) -> list:
        return [self]

    def differentiate(self, param):
        if param is self:
            return Constant(1)
        return Constant(0)

    def __repr__(self):
        return f"Parameter('{self.name}', prior={self.prior})"


class ASC(Parameter):
    """Alternative-specific constant.

    Identical to Parameter but semantically tagged as a constant
    term (not multiplied by an attribute). The prior should reflect
    the expected market share of the alternative.

    Parameters
    ----------
    name : str
        Constant name.
    prior : float
        Prior value.
    """

    def __init__(self, name: str, prior: float):
        super().__init__(name, prior)

    def evaluate(self, data=None, draws=None):
        return np.float64(self.prior)

    def __repr__(self):
        return f"ASC('{self.name}', prior={self.prior})"


# ---------------------------------------------------------------------------
# Binary operations
# ---------------------------------------------------------------------------

class Add(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, data=None, draws=None):
        return self.left.evaluate(data, draws) + self.right.evaluate(data, draws)

    def get_params(self):
        return self.left.get_params() + self.right.get_params()

    def differentiate(self, param):
        return Add(self.left.differentiate(param), self.right.differentiate(param))

    def __repr__(self):
        return f"({self.left!r} + {self.right!r})"


class Sub(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, data=None, draws=None):
        return self.left.evaluate(data, draws) - self.right.evaluate(data, draws)

    def get_params(self):
        return self.left.get_params() + self.right.get_params()

    def differentiate(self, param):
        return Sub(self.left.differentiate(param), self.right.differentiate(param))

    def __repr__(self):
        return f"({self.left!r} - {self.right!r})"


class Mul(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, data=None, draws=None):
        return self.left.evaluate(data, draws) * self.right.evaluate(data, draws)

    def get_params(self):
        return self.left.get_params() + self.right.get_params()

    def differentiate(self, param):
        # Product rule: (f·g)' = f'·g + f·g'
        return Add(
            Mul(self.left.differentiate(param), self.right),
            Mul(self.left, self.right.differentiate(param)),
        )

    def __repr__(self):
        return f"({self.left!r} * {self.right!r})"


class Div(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def evaluate(self, data=None, draws=None):
        return self.left.evaluate(data, draws) / self.right.evaluate(data, draws)

    def get_params(self):
        return self.left.get_params() + self.right.get_params()

    def differentiate(self, param):
        # Quotient rule: (f/g)' = (f'g − fg') / g²
        return Div(
            Sub(
                Mul(self.left.differentiate(param), self.right),
                Mul(self.left, self.right.differentiate(param)),
            ),
            Pow(self.right, Constant(2)),
        )

    def __repr__(self):
        return f"({self.left!r} / {self.right!r})"


class Pow(Expression):
    def __init__(self, base: Expression, exponent: Expression):
        self.base = base
        self.exponent = exponent

    def evaluate(self, data=None, draws=None):
        return self.base.evaluate(data, draws) ** self.exponent.evaluate(data, draws)

    def get_params(self):
        return self.base.get_params() + self.exponent.get_params()

    def differentiate(self, param):
        # General power rule: d/dx [f^g] = f^g · (g'·ln(f) + g·f'/f)
        return Mul(
            Pow(self.base, self.exponent),
            Add(
                Mul(self.exponent.differentiate(param), Log(self.base)),
                Mul(self.exponent, Div(self.base.differentiate(param), self.base)),
            ),
        )

    def __repr__(self):
        return f"({self.base!r} ** {self.exponent!r})"


class Neg(Expression):
    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data=None, draws=None):
        return -self.expr.evaluate(data, draws)

    def get_params(self):
        return self.expr.get_params()

    def differentiate(self, param):
        return Neg(self.expr.differentiate(param))

    def __repr__(self):
        return f"(-{self.expr!r})"


# ---------------------------------------------------------------------------
# Comparison expressions (indicator: 1.0 where condition holds, 0.0 otherwise)
# ---------------------------------------------------------------------------

class _Comparison(Expression):
    """Base for binary comparison expressions."""

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def get_params(self):
        return self.left.get_params() + self.right.get_params()

    # Comparisons are step functions — derivative is 0 w.r.t. any parameter.
    def differentiate(self, param):
        return Constant(0)


class Equal(_Comparison):
    def evaluate(self, data=None, draws=None):
        return (self.left.evaluate(data, draws) == self.right.evaluate(data, draws)).astype(float)

    def __repr__(self):
        return f"({self.left!r} == {self.right!r})"


class NotEqual(_Comparison):
    def evaluate(self, data=None, draws=None):
        return (self.left.evaluate(data, draws) != self.right.evaluate(data, draws)).astype(float)

    def __repr__(self):
        return f"({self.left!r} != {self.right!r})"


class LessThan(_Comparison):
    def evaluate(self, data=None, draws=None):
        return (self.left.evaluate(data, draws) < self.right.evaluate(data, draws)).astype(float)

    def __repr__(self):
        return f"({self.left!r} < {self.right!r})"


class LessEqual(_Comparison):
    def evaluate(self, data=None, draws=None):
        return (self.left.evaluate(data, draws) <= self.right.evaluate(data, draws)).astype(float)

    def __repr__(self):
        return f"({self.left!r} <= {self.right!r})"


class GreaterThan(_Comparison):
    def evaluate(self, data=None, draws=None):
        return (self.left.evaluate(data, draws) > self.right.evaluate(data, draws)).astype(float)

    def __repr__(self):
        return f"({self.left!r} > {self.right!r})"


class GreaterEqual(_Comparison):
    def evaluate(self, data=None, draws=None):
        return (self.left.evaluate(data, draws) >= self.right.evaluate(data, draws)).astype(float)

    def __repr__(self):
        return f"({self.left!r} >= {self.right!r})"


# ---------------------------------------------------------------------------
# Unary functions
# ---------------------------------------------------------------------------

class Exp(Expression):
    """Element-wise exponential."""

    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data=None, draws=None):
        return np.exp(self.expr.evaluate(data, draws))

    def get_params(self):
        return self.expr.get_params()

    def differentiate(self, param):
        # d/dx exp(f) = exp(f)·f'
        return Mul(Exp(self.expr), self.expr.differentiate(param))

    def __repr__(self):
        return f"exp({self.expr!r})"


class Log(Expression):
    """Element-wise natural logarithm."""

    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data=None, draws=None):
        return np.log(self.expr.evaluate(data, draws))

    def get_params(self):
        return self.expr.get_params()

    def differentiate(self, param):
        # d/dx ln(f) = f'/f
        return Div(self.expr.differentiate(param), self.expr)

    def __repr__(self):
        return f"log({self.expr!r})"


# ---------------------------------------------------------------------------
# Monte Carlo wrapper (placeholder for Bayesian designs, Phase 3)
# ---------------------------------------------------------------------------

class MonteCarlo(Expression):
    """Averages an expression over Monte Carlo draws.

    When ``draws`` is provided and the inner expression returns a
    2-D array of shape ``(draws, ncs)``, this wrapper collapses the
    draw dimension by taking the mean. For non-random expressions
    (draws dimension absent) it is a no-op.
    """

    def __init__(self, expr: Expression):
        self.expr = expr

    def evaluate(self, data=None, draws=None):
        val = self.expr.evaluate(data, draws)
        if isinstance(val, np.ndarray) and val.ndim == 2:
            return val.mean(axis=0)
        return val

    def get_params(self):
        return self.expr.get_params()

    def differentiate(self, param):
        return MonteCarlo(self.expr.differentiate(param))

    def __repr__(self):
        return f"MonteCarlo({self.expr!r})"


# ---------------------------------------------------------------------------
# Free functions (module-level, mirror Biogeme's API)
# ---------------------------------------------------------------------------

def exp(expr) -> Exp:
    """Natural exponential of an expression."""
    return Exp(_wrap(expr))


def log(expr) -> Log:
    """Natural logarithm of an expression."""
    return Log(_wrap(expr))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def get_unique_params(V: dict) -> List[Parameter]:
    """Collect all unique Parameter objects from a utility-function dict.

    Parameters
    ----------
    V : dict
        Dictionary mapping alternative indices to Expression objects,
        e.g. ``{1: V1, 2: V2}``.

    Returns
    -------
    list[Parameter]
        Ordered list of unique Parameter objects (by object identity),
        preserving first-seen order.
    """
    seen_ids = set()
    params = []
    for expr in V.values():
        for p in expr.get_params():
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                params.append(p)
    return params
