"""Unit tests for the native expression system."""

import numpy as np
import pandas as pd
import pytest
from choicedesign.expressions import (
    Constant, Attribute, Parameter, ASC,
    Add, Sub, Mul, Div, Pow, Neg,
    Exp, Log, MonteCarlo,
    exp, log, get_unique_params,
)

DATA = pd.DataFrame({'x': [1.0, 2.0, 3.0], 'y': [3.0, 2.0, 1.0]})


# ---------------------------------------------------------------------------
# Leaf nodes
# ---------------------------------------------------------------------------

def test_constant_evaluate():
    assert Constant(2.5).evaluate() == pytest.approx(2.5)
    assert Constant(0).evaluate() == pytest.approx(0.0)

def test_constant_differentiate():
    p = Parameter('b', 1.0)
    assert Constant(5).differentiate(p).evaluate() == pytest.approx(0.0)

def test_parameter_evaluate():
    p = Parameter('beta', -0.1)
    assert p.evaluate() == pytest.approx(-0.1)

def test_parameter_get_params():
    p = Parameter('beta', -0.1)
    assert p.get_params() == [p]

def test_parameter_differentiate_self():
    p = Parameter('beta', -0.1)
    assert p.differentiate(p).evaluate() == pytest.approx(1.0)

def test_parameter_differentiate_other():
    p = Parameter('beta', -0.1)
    q = Parameter('gamma', 0.5)
    assert p.differentiate(q).evaluate() == pytest.approx(0.0)

def test_asc_evaluate():
    a = ASC('asc_1', 0.5)
    assert a.evaluate() == pytest.approx(0.5)

def test_asc_is_parameter_subclass():
    a = ASC('asc_1', 0.5)
    assert isinstance(a, Parameter)

def test_attribute_evaluate():
    attr = Attribute('x', [1, 2, 3])
    result = attr.evaluate(DATA)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

def test_attribute_evaluate_no_data_raises():
    attr = Attribute('x', [1, 2, 3])
    with pytest.raises(ValueError):
        attr.evaluate()

def test_attribute_levels_from_int():
    attr = Attribute('x', 3)
    assert attr.levels == [0, 1, 2]

def test_attribute_differentiate():
    p = Parameter('beta', 1.0)
    attr = Attribute('x', [1, 2, 3])
    result = attr.differentiate(p).evaluate(DATA)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------

def test_add():
    p = Parameter('b', 2.0)
    c = Constant(3.0)
    np.testing.assert_allclose((p + c).evaluate(), 5.0)

def test_sub():
    p = Parameter('b', 5.0)
    c = Constant(3.0)
    np.testing.assert_allclose((p - c).evaluate(), 2.0)

def test_mul_scalar():
    p = Parameter('b', -0.1)
    attr = Attribute('x', [1, 2, 3])
    result = (p * attr).evaluate(DATA)
    np.testing.assert_allclose(result, [-0.1, -0.2, -0.3])

def test_div():
    np.testing.assert_allclose((Constant(6.0) / Constant(2.0)).evaluate(), 3.0)

def test_pow():
    np.testing.assert_allclose((Constant(2.0) ** Constant(3.0)).evaluate(), 8.0)

def test_neg():
    p = Parameter('b', 0.5)
    np.testing.assert_allclose((-p).evaluate(), -0.5)

def test_radd():
    p = Parameter('b', 2.0)
    np.testing.assert_allclose((1 + p).evaluate(), 3.0)

def test_rsub():
    p = Parameter('b', 1.0)
    np.testing.assert_allclose((5 - p).evaluate(), 4.0)

def test_rmul():
    p = Parameter('b', 3.0)
    np.testing.assert_allclose((2 * p).evaluate(), 6.0)


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

def test_equal():
    attr = Attribute('x', [1, 2, 3])
    result = (attr == 2).evaluate(DATA)
    np.testing.assert_array_equal(result, [0.0, 1.0, 0.0])

def test_not_equal():
    attr = Attribute('x', [1, 2, 3])
    result = (attr != 2).evaluate(DATA)
    np.testing.assert_array_equal(result, [1.0, 0.0, 1.0])

def test_less_than():
    attr = Attribute('x', [1, 2, 3])
    result = (attr < 2).evaluate(DATA)
    np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

def test_less_equal():
    attr = Attribute('x', [1, 2, 3])
    result = (attr <= 2).evaluate(DATA)
    np.testing.assert_array_equal(result, [1.0, 1.0, 0.0])

def test_greater_than():
    attr = Attribute('x', [1, 2, 3])
    result = (attr > 2).evaluate(DATA)
    np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

def test_greater_equal():
    attr = Attribute('x', [1, 2, 3])
    result = (attr >= 2).evaluate(DATA)
    np.testing.assert_array_equal(result, [0.0, 1.0, 1.0])

def test_comparison_returns_float():
    attr = Attribute('x', [1, 2, 3])
    result = (attr == 1).evaluate(DATA)
    assert result.dtype == float

def test_comparison_differentiate_is_zero():
    p = Parameter('b', 1.0)
    attr = Attribute('x', [1, 2, 3])
    result = (attr == 2).differentiate(p).evaluate(DATA)
    assert result == pytest.approx(0.0)

def test_constant_equality_scalar_path():
    # Both sides scalar — exercises numpy.bool_ .astype(float) path
    result = (Constant(1) == Constant(1)).evaluate()
    assert result == pytest.approx(1.0)
    result = (Constant(1) == Constant(2)).evaluate()
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unary functions
# ---------------------------------------------------------------------------

def test_exp_evaluate():
    p = Parameter('b', 1.0)
    np.testing.assert_allclose(Exp(p).evaluate(), np.e)

def test_log_evaluate():
    np.testing.assert_allclose(Log(Constant(np.e)).evaluate(), 1.0)

def test_exp_free_function():
    p = Parameter('b', 0.0)
    np.testing.assert_allclose(exp(p).evaluate(), 1.0)

def test_log_free_function():
    np.testing.assert_allclose(log(Constant(1.0)).evaluate(), 0.0)

def test_exp_differentiate():
    # d/db exp(b) = exp(b)
    p = Parameter('b', 0.0)
    deriv = Exp(p).differentiate(p)
    np.testing.assert_allclose(deriv.evaluate(), 1.0)  # exp(0) * 1

def test_log_differentiate():
    # d/db log(b) = 1/b  → at b=2: 0.5
    p = Parameter('b', 2.0)
    deriv = Log(p).differentiate(p)
    np.testing.assert_allclose(deriv.evaluate(), 0.5)


# ---------------------------------------------------------------------------
# Symbolic differentiation
# ---------------------------------------------------------------------------

def test_differentiate_linear():
    # d(beta * x)/d(beta) = x
    p = Parameter('beta', -0.1)
    attr = Attribute('x', [1, 2, 3])
    deriv = (p * attr).differentiate(p)
    np.testing.assert_allclose(deriv.evaluate(DATA), [1.0, 2.0, 3.0])

def test_differentiate_sum():
    # d(beta*x + gamma*y)/d(beta) = x
    p = Parameter('beta', 1.0)
    q = Parameter('gamma', 2.0)
    attr_x = Attribute('x', [1, 2, 3])
    attr_y = Attribute('y', [3, 2, 1])
    expr = p * attr_x + q * attr_y
    deriv = expr.differentiate(p)
    np.testing.assert_allclose(deriv.evaluate(DATA), [1.0, 2.0, 3.0])

def test_differentiate_unrelated_param_is_zero():
    p = Parameter('beta', 1.0)
    q = Parameter('gamma', 2.0)
    attr = Attribute('x', [1, 2, 3])
    deriv = (p * attr).differentiate(q)
    np.testing.assert_allclose(deriv.evaluate(DATA), [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# get_params / get_unique_params
# ---------------------------------------------------------------------------

def test_get_params_deduplication():
    p = Parameter('beta', 1.0)
    attr = Attribute('x', [1, 2, 3])
    # beta appears twice in beta*x + beta*x
    expr = p * attr + p * attr
    raw = expr.get_params()
    assert raw.count(p) == 2

def test_get_unique_params_deduplication():
    p = Parameter('beta', 1.0)
    q = Parameter('gamma', 2.0)
    attr = Attribute('x', [1, 2, 3])
    V = {1: p * attr + q * attr, 2: p * attr}
    unique = get_unique_params(V)
    assert unique == [p, q]

def test_get_unique_params_asc_excluded_by_isinstance():
    # ASC is a Parameter subclass — get_unique_params includes it;
    # exclusion from D-error happens in _derr, not here.
    asc = ASC('asc_1', 0.5)
    p = Parameter('beta', 1.0)
    attr = Attribute('x', [1, 2, 3])
    V = {1: asc + p * attr, 2: p * attr}
    unique = get_unique_params(V)
    assert asc in unique
    assert p in unique


# ---------------------------------------------------------------------------
# MonteCarlo wrapper
# ---------------------------------------------------------------------------

def test_montecarlo_passthrough_1d():
    # For a non-random (1-D) expression, MonteCarlo is a no-op
    p = Parameter('b', 3.0)
    mc = MonteCarlo(p)
    assert mc.evaluate() == pytest.approx(3.0)

def test_montecarlo_collapses_2d():
    # Simulate a 2-D draws × ncs array
    class FakeExpr:
        def evaluate(self, data=None, draws=None):
            return np.array([[1.0, 2.0], [3.0, 4.0]])  # (2 draws, 2 cs)
        def get_params(self): return []
        def differentiate(self, p): return Constant(0)

    mc = MonteCarlo(FakeExpr())
    result = mc.evaluate()
    np.testing.assert_allclose(result, [2.0, 3.0])  # mean over draw axis
