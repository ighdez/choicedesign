Getting Started
===============

Installation
------------

Install ChoiceDesign from PyPI using pip::

    pip install choicedesign

Or, to install from source with Poetry::

    git clone https://github.com/ighdez/choicedesign.git
    cd choicedesign
    poetry install

Basic workflow
--------------

A ChoiceDesign session has six steps.

**1. Define attributes**

:class:`~choicedesign.expressions.Attribute` objects represent the columns of the
design matrix.  Each attribute has a name and a list of discrete levels::

    from choicedesign.expressions import Attribute, Parameter, ASC

    alt1_cost = Attribute('alt1_cost', [100, 200, 300])
    alt2_cost = Attribute('alt2_cost', [100, 200, 300])
    alt1_time = Attribute('alt1_time', [10, 20, 30])
    alt2_time = Attribute('alt2_time', [10, 20, 30])

**2. Define parameters**

:class:`~choicedesign.expressions.Parameter` objects carry a prior value used
to evaluate the D-error.  For models with alternative-specific constants use
:class:`~choicedesign.expressions.ASC` — these are automatically excluded from
the D-error computation::

    beta_cost = Parameter('beta_cost', -0.01)
    beta_time = Parameter('beta_time', -0.05)
    asc_1     = ASC('asc_1', 0.5)

**3. Write utility functions**

Utility functions are built by composing expressions with ordinary Python
arithmetic.  The result is passed to :meth:`~choicedesign.design.EffDesign.optimise`
as a dictionary keyed by alternative index::

    V1 = asc_1 + beta_cost * alt1_cost + beta_time * alt1_time
    V2 =         beta_cost * alt2_cost + beta_time * alt2_time

**4. Create the design object and generate an initial design**

:class:`~choicedesign.design.EffDesign` holds the design configuration.
:meth:`~choicedesign.design.EffDesign.gen_initdesign` generates a random
balanced starting point.  Optionally pass conditions that every row must
satisfy (see :doc:`concepts` for the condition syntax)::

    from choicedesign.design import EffDesign

    design = EffDesign(X=[alt1_cost, alt2_cost, alt1_time, alt2_time], ncs=18)
    init   = design.gen_initdesign(
        cond=['alt1_cost > alt2_cost'],
        seed=42,
    )

**5. Optimise**

At least one stopping criterion is required.  The algorithm keeps the best
design seen so far and prints live progress when ``verbose=True``::

    result = design.optimise(
        init,
        V={1: V1, 2: V2},
        time_lim=2,       # stop after 2 minutes
        verbose=True,
    )

    optimal_design, init_derr, final_derr, n_iter, util_balance = result

The return values are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Description
   * - ``optimal_design``
     - :class:`pandas.DataFrame` with a ``CS`` column and one column per attribute.
   * - ``init_derr``
     - Criterion value of the initial (random) design.
   * - ``final_derr``
     - Criterion value of the optimised design.
   * - ``n_iter``
     - Total number of algorithm iterations.
   * - ``util_balance``
     - Utility balance ratio (100 % = equal expected shares across alternatives).

**6. Optional: blocking**

:meth:`~choicedesign.design.EffDesign.gen_blocks` partitions the choice
situations into blocks while minimising the correlation between the block
column and all attributes::

    blocked_design, corr_history = design.gen_blocks(optimal_design, n_blocks=3)

Evaluating an existing design
------------------------------

:meth:`~choicedesign.design.EffDesign.evaluate` computes the criterion value
and utility balance for any design stored in a DataFrame::

    d_error, ub = design.evaluate(optimal_design, V={1: V1, 2: V2})
