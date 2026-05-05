"""End-to-end smoke tests based on the example notebooks."""

import numpy as np
import pytest
from choicedesign.design import EffDesign
from choicedesign.expressions import Attribute, Parameter, ASC

TIME_LIM = 0.05  # ~3 seconds per test


def _assert_valid(init_perf, final_perf, ubalance_ratio):
    assert final_perf < np.inf, "D-error is infinite (singular information matrix)"
    assert final_perf <= init_perf, "Optimiser did not improve the design"
    assert ubalance_ratio > 0, "Utility balance ratio should be positive"


def test_smoke_simple():
    """Linear MNL, 2 alternatives, 4 continuous attributes (rum_simple)."""
    alt1_A = Attribute('alt1_A', [1, 2, 3])
    alt1_B = Attribute('alt1_B', [10, 15, 15.5])
    alt1_C = Attribute('alt1_C', [0, 3, 5])
    alt1_D = Attribute('alt1_D', [0, 1, 2])
    alt2_A = Attribute('alt2_A', [1, 2, 3])
    alt2_B = Attribute('alt2_B', [10, 15, 15.5])
    alt2_C = Attribute('alt2_C', [0, 3, 5])
    alt2_D = Attribute('alt2_D', [0, 1, 2])

    beta_A = Parameter('beta_A', -0.1)
    beta_B = Parameter('beta_B', -0.02)
    beta_C = Parameter('beta_C', 0.1)
    beta_D = Parameter('beta_D', 0.15)

    V = {
        1: beta_A * alt1_A + beta_B * alt1_B + beta_C * alt1_C + beta_D * alt1_D,
        2: beta_A * alt2_A + beta_B * alt2_B + beta_C * alt2_C + beta_D * alt2_D,
    }

    design = EffDesign(X=[alt1_A, alt1_B, alt1_C, alt1_D, alt2_A, alt2_B, alt2_C, alt2_D], ncs=18)
    init = design.gen_initdesign(seed=42)

    optimal, init_perf, final_perf, _, ubalance_ratio = design.optimise(init, V, time_lim=TIME_LIM)
    _assert_valid(init_perf, final_perf, ubalance_ratio)

    eval_perf, eval_ub = design.evaluate(optimal, V)
    assert abs(eval_perf - final_perf) < 1e-10, "evaluate() D-error differs from optimise()"

    blocked, corr = design.gen_blocks(optimal, n_blocks=3)
    assert 'Block' in blocked.columns
    assert set(blocked['Block'].unique()) == {1, 2, 3}


def test_smoke_dummy():
    """Dummy-coded categorical attributes (rum_dummy)."""
    alt1_A = Attribute('alt1_A', [1, 2, 3])
    alt1_B = Attribute('alt1_B', [1, 2, 3])
    alt1_C = Attribute('alt1_C', [0, 3, 5])
    alt1_D = Attribute('alt1_D', [0, 1, 2])
    alt2_A = Attribute('alt2_A', [1, 2, 3])
    alt2_B = Attribute('alt2_B', [1, 2, 3])
    alt2_C = Attribute('alt2_C', [0, 3, 5])
    alt2_D = Attribute('alt2_D', [0, 1, 2])

    beta_A_2 = Parameter('beta_A_2', -0.1)
    beta_A_3 = Parameter('beta_A_3', -0.4)
    beta_B_2 = Parameter('beta_B_2', -0.02)
    beta_B_3 = Parameter('beta_B_3', -0.01)
    beta_C   = Parameter('beta_C', 0.1)
    beta_D   = Parameter('beta_D', 0.15)

    V = {
        1: beta_A_2 * (alt1_A == 2) + beta_A_3 * (alt1_A == 3)
           + beta_B_2 * (alt1_B == 2) + beta_B_3 * (alt1_B == 3)
           + beta_C * alt1_C + beta_D * alt1_D,
        2: beta_A_2 * (alt2_A == 2) + beta_A_3 * (alt2_A == 3)
           + beta_B_2 * (alt2_B == 2) + beta_B_3 * (alt2_B == 3)
           + beta_C * alt2_C + beta_D * alt2_D,
    }

    design = EffDesign(X=[alt1_A, alt1_B, alt1_C, alt1_D, alt2_A, alt2_B, alt2_C, alt2_D], ncs=18)
    init = design.gen_initdesign(seed=42)

    optimal, init_perf, final_perf, _, ubalance_ratio = design.optimise(init, V, time_lim=TIME_LIM)
    _assert_valid(init_perf, final_perf, ubalance_ratio)

    eval_perf, _ = design.evaluate(optimal, V)
    assert abs(eval_perf - final_perf) < 1e-10, "evaluate() D-error differs from optimise()"


def test_smoke_conditions():
    """Conditions on attributes — verifies the optimised design satisfies constraints (rum_conds)."""
    alt1_A = Attribute('alt1_A', [1, 2, 3])
    alt1_B = Attribute('alt1_B', [10, 15, 15.5])
    alt1_C = Attribute('alt1_C', [0, 3, 5])
    alt1_D = Attribute('alt1_D', [0, 1, 2])
    alt2_A = Attribute('alt2_A', [1, 2, 3])
    alt2_B = Attribute('alt2_B', [10, 15, 15.5])
    alt2_C = Attribute('alt2_C', [0, 3, 5])
    alt2_D = Attribute('alt2_D', [0, 1, 2])

    cond = [
        'alt1_A > alt2_A',
        'if alt1_B > 10 then alt2_A < 3',
        'if alt1_A > 1 then alt2_A < 3',
    ]

    beta_A = Parameter('beta_A', -0.1)
    beta_B = Parameter('beta_B', -0.02)
    beta_C = Parameter('beta_C', 0.1)
    beta_D = Parameter('beta_D', 0.15)

    V = {
        1: beta_A * alt1_A + beta_B * alt1_B + beta_C * alt1_C + beta_D * alt1_D,
        2: beta_A * alt2_A + beta_B * alt2_B + beta_C * alt2_C + beta_D * alt2_D,
    }

    design = EffDesign(X=[alt1_A, alt1_B, alt1_C, alt1_D, alt2_A, alt2_B, alt2_C, alt2_D], ncs=18)
    init = design.gen_initdesign(cond=cond, seed=42)

    # Condition 1 must hold in the initial design
    assert (init['alt1_A'] > init['alt2_A']).all(), "Condition alt1_A > alt2_A violated in init design"

    optimal, init_perf, final_perf, _, ubalance_ratio = design.optimise(init, V, time_lim=TIME_LIM)
    _assert_valid(init_perf, final_perf, ubalance_ratio)

    opt_data = optimal.drop('CS', axis=1)
    assert (opt_data['alt1_A'] > opt_data['alt2_A']).all(), "Condition alt1_A > alt2_A violated in optimal design"


def test_smoke_optout():
    """3-alternative design with a constant opt-out utility (rum_optout)."""
    alt1_A = Attribute('alt1_A', [1, 2, 3])
    alt1_B = Attribute('alt1_B', [10, 15, 15.5])
    alt1_C = Attribute('alt1_C', [0, 3, 5])
    alt1_D = Attribute('alt1_D', [0, 1, 2])
    alt2_A = Attribute('alt2_A', [1, 2, 3])
    alt2_B = Attribute('alt2_B', [10, 15, 15.5])
    alt2_C = Attribute('alt2_C', [0, 3, 5])
    alt2_D = Attribute('alt2_D', [0, 1, 2])

    asc_optout = ASC('asc_optout', -1)
    beta_A = Parameter('beta_A', -0.1)
    beta_B = Parameter('beta_B', -0.02)
    beta_C = Parameter('beta_C', 0.1)
    beta_D = Parameter('beta_D', 0.15)

    V = {
        1: beta_A * alt1_A + beta_B * alt1_B + beta_C * alt1_C + beta_D * alt1_D,
        2: beta_A * alt2_A + beta_B * alt2_B + beta_C * alt2_C + beta_D * alt2_D,
        3: asc_optout,
    }

    design = EffDesign(X=[alt1_A, alt1_B, alt1_C, alt1_D, alt2_A, alt2_B, alt2_C, alt2_D], ncs=18)
    init = design.gen_initdesign(seed=42)

    optimal, init_perf, final_perf, _, ubalance_ratio = design.optimise(init, V, time_lim=TIME_LIM)
    _assert_valid(init_perf, final_perf, ubalance_ratio)

    eval_perf, _ = design.evaluate(optimal, V)
    assert abs(eval_perf - final_perf) < 1e-10, "evaluate() D-error differs from optimise()"


def test_smoke_asc():
    """Linear MNL with an ASC — ASC must be excluded from D-error count (rum_asc)."""
    alt1_A = Attribute('alt1_A', [1, 2, 3])
    alt1_B = Attribute('alt1_B', [10, 15, 15.5])
    alt1_C = Attribute('alt1_C', [0, 3, 5])
    alt1_D = Attribute('alt1_D', [0, 1, 2])
    alt2_A = Attribute('alt2_A', [1, 2, 3])
    alt2_B = Attribute('alt2_B', [10, 15, 15.5])
    alt2_C = Attribute('alt2_C', [0, 3, 5])
    alt2_D = Attribute('alt2_D', [0, 1, 2])

    asc_1  = ASC('asc_1', 0.5)
    beta_A = Parameter('beta_A', -0.1)
    beta_B = Parameter('beta_B', -0.02)
    beta_C = Parameter('beta_C', 0.1)
    beta_D = Parameter('beta_D', 0.15)

    V = {
        1: asc_1 + beta_A * alt1_A + beta_B * alt1_B + beta_C * alt1_C + beta_D * alt1_D,
        2:         beta_A * alt2_A + beta_B * alt2_B + beta_C * alt2_C + beta_D * alt2_D,
    }

    design = EffDesign(X=[alt1_A, alt1_B, alt1_C, alt1_D, alt2_A, alt2_B, alt2_C, alt2_D], ncs=18)
    init = design.gen_initdesign(seed=42)

    optimal, init_perf, final_perf, _, ubalance_ratio = design.optimise(init, V, time_lim=TIME_LIM)
    _assert_valid(init_perf, final_perf, ubalance_ratio)

    eval_perf, _ = design.evaluate(optimal, V)
    assert abs(eval_perf - final_perf) < 1e-10, "evaluate() D-error differs from optimise()"
