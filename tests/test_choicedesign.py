import numpy as np
import pytest

from choicedesign import __version__
from choicedesign.utils import _parse_condition
from choicedesign.design import EffDesign
from choicedesign.expressions import Attribute, Parameter


def test_version():
    assert __version__ == '1.0.0'


# ---------------------------------------------------------------------------
# _parse_condition unit tests
# ---------------------------------------------------------------------------

NAMES = ['alt1_A', 'alt1_B', 'alt2_A', 'alt2_B']

# 3 rows, 4 columns
DESMAT = np.array([
    [1, 10,   2, 15  ],   # row 0: alt1_A < alt2_A
    [3, 15,   1, 10  ],   # row 1: alt1_A > alt2_A, alt1_B > 10
    [2, 15.5, 1, 10  ],   # row 2: alt1_A > alt2_A, alt1_B > 10
], dtype=float)


def test_parse_condition_binary_attributes():
    f = _parse_condition('alt1_A > alt2_A', NAMES)
    result = f(DESMAT)
    np.testing.assert_array_equal(result, [False, True, True])


def test_parse_condition_binary_numeric():
    f = _parse_condition('alt1_B > 10', NAMES)
    result = f(DESMAT)
    np.testing.assert_array_equal(result, [False, True, True])


def test_parse_condition_ifthen():
    # if alt1_B > 10 then alt2_A < 3  =>  NOT(alt1_B > 10) OR alt2_A < 3
    # row 0: NOT(False) OR True  = True
    # row 1: NOT(True)  OR True  = True
    # row 2: NOT(True)  OR True  = True
    f = _parse_condition('if alt1_B > 10 then alt2_A < 3', NAMES)
    result = f(DESMAT)
    np.testing.assert_array_equal(result, [True, True, True])


def test_parse_condition_ifthen_violated():
    # if alt1_A > 1 then alt2_A < 2  =>  NOT(alt1_A > 1) OR alt2_A < 2
    # row 0: NOT(False) OR False = True   (antecedent false -> implication holds)
    # row 1: NOT(True)  OR True  = True
    # row 2: NOT(True)  OR True  = True
    desmat = np.array([
        [1, 10, 3, 15],   # alt2_A=3, but alt1_A=1 so antecedent is false
        [3, 15, 1, 10],
        [2, 15, 1, 10],
    ], dtype=float)
    f = _parse_condition('if alt1_A > 1 then alt2_A < 2', NAMES)
    result = f(desmat)
    np.testing.assert_array_equal(result, [True, True, True])

    # Now a row that actually violates: alt1_A=3 (>1) and alt2_A=3 (not <2)
    desmat_bad = np.array([[3, 10, 3, 15]], dtype=float)
    assert not np.all(f(desmat_bad))


def test_parse_condition_compound_and():
    # alt1_A > alt2_A & alt1_B > 10
    f = _parse_condition('alt1_A > alt2_A & alt1_B > 10', NAMES)
    result = f(DESMAT)
    np.testing.assert_array_equal(result, [False, True, True])


def test_parse_condition_single_row_slice():
    f = _parse_condition('alt1_A > alt2_A', NAMES)
    assert np.all(f(DESMAT[1:2, :]))   # row 1 satisfies
    assert not np.all(f(DESMAT[0:1, :]))  # row 0 does not


def test_parse_condition_unknown_attribute_raises():
    with pytest.raises(ValueError, match="Unknown attribute 'al1t_A'"):
        _parse_condition('al1t_A > alt2_A', NAMES)


def test_parse_condition_unknown_rhs_raises():
    with pytest.raises(ValueError, match="not a known attribute"):
        _parse_condition('alt1_A > unknown_col', NAMES)


def test_parse_condition_malformed_raises():
    with pytest.raises(ValueError, match="Cannot parse condition"):
        _parse_condition('alt1_A', NAMES)


# ---------------------------------------------------------------------------
# Integration tests: gen_initdesign and optimise respect conditions
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_design():
    alt1_A = Attribute('alt1_A', [1, 2, 3])
    alt1_B = Attribute('alt1_B', [10, 15, 15.5])
    alt2_A = Attribute('alt2_A', [1, 2, 3])
    alt2_B = Attribute('alt2_B', [10, 15, 15.5])
    return EffDesign(X=[alt1_A, alt1_B, alt2_A, alt2_B], ncs=12), alt1_A, alt1_B, alt2_A, alt2_B


def test_initdesign_respects_binary_condition(simple_design):
    design, *_ = simple_design
    init = design.gen_initdesign(cond=['alt1_A > alt2_A'], seed=0)
    desmat = init.to_numpy(dtype=float)
    assert np.all(desmat[:, 0] > desmat[:, 2])


def test_initdesign_respects_ifthen_condition(simple_design):
    design, *_ = simple_design
    init = design.gen_initdesign(cond=['if alt1_B > 10 then alt2_A < 3'], seed=0)
    desmat = init.to_numpy(dtype=float)
    # where alt1_B > 10, alt2_A must be < 3
    mask = desmat[:, 1] > 10
    assert np.all(desmat[mask, 2] < 3)


def test_initdesign_bad_attr_raises(simple_design):
    design, *_ = simple_design
    with pytest.raises(ValueError, match="Unknown attribute"):
        design.gen_initdesign(cond=['al1t_A > alt2_A'])


def test_optimise_respects_conditions(simple_design):
    design, alt1_A, alt1_B, alt2_A, alt2_B = simple_design
    init = design.gen_initdesign(cond=['alt1_A > alt2_A'], seed=0)

    beta_A = Parameter('beta_A', -0.1)
    beta_B = Parameter('beta_B', -0.02)
    V = {1: beta_A * alt1_A + beta_B * alt1_B,
         2: beta_A * alt2_A + beta_B * alt2_B}

    result = design.optimise(init, V=V, time_lim=0.05)
    final = result[0].drop('CS', axis=1).to_numpy(dtype=float)
    assert np.all(final[:, 0] > final[:, 2])
