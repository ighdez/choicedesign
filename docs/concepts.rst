Concepts
========

Discrete choice experiments
----------------------------

A **discrete choice experiment (DCE)** presents respondents with a series of
choice situations, each offering two or more alternatives described by
attribute levels (e.g. cost, travel time, comfort).  The analyst's goal is to
estimate how much respondents value each attribute — the model parameters.

The quality of parameter estimates depends heavily on the design of these
choice situations.  A poor design can produce correlated or unidentifiable
parameters; a good design minimises estimation variance.

D-error and optimality criteria
---------------------------------

ChoiceDesign minimises a scalar measure of design quality derived from the
**Fisher information matrix** :math:`I(\beta)`.

For the multinomial logit (MNL) model, the information matrix entry for
parameters :math:`k` and :math:`l` is:

.. math::

   I_{kl} = \sum_n \left[
       \sum_j P_{nj}\, x_{njk}\, x_{njl}
       - \left(\sum_j P_{nj}\, x_{njk}\right)
         \left(\sum_j P_{nj}\, x_{njl}\right)
   \right]

where :math:`P_{nj}` is the MNL choice probability for alternative *j* in
choice situation *n*, and :math:`x_{njk} = \partial V_j / \partial \beta_k`.

Three optimality criteria are supported:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Criterion
     - Argument
     - Definition
   * - D-error
     - ``criterion='d'``
     - :math:`\det\!\left(I^{-1}\right)^{1/K}` — minimises the generalised
       variance of all *K* non-ASC parameter estimates.
   * - A-error
     - ``criterion='a'``
     - :math:`\operatorname{trace}(I^{-1}) / K` — minimises the average
       parameter variance.
   * - C-error
     - ``criterion='c'``
     - Sum of WTP variances computed via the delta method.  Requires
       ``cost_param`` and ``wtp_params``.

A lower value is always better.  The algorithm returns ``np.inf`` when the
information matrix is singular (the design is not identified).

**Bayesian (Db-efficient) designs**

When parameters are uncertain, set ``prior_std`` on a
:class:`~choicedesign.expressions.Parameter` and pass ``bayes_draws`` to
:meth:`~choicedesign.design.EffDesign.optimise`.  The Db-error is the
expected D-error averaged over Monte Carlo draws from
:math:`\beta_k \sim \mathcal{N}(\text{prior}, \text{prior\_std}^2)`.

The expression system
----------------------

Utility functions are built from a tree of
:class:`~choicedesign.expressions.Expression` nodes.  Python's arithmetic
operators are overloaded so utilities look like standard equations::

    V1 = asc_1 + beta_cost * alt1_cost + beta_time * alt1_time

Every node supports **symbolic differentiation** via
:meth:`~choicedesign.expressions.Expression.differentiate`.
:class:`~choicedesign.criteria.MNLModel` calls this once at construction time
to pre-compile the gradient tensor
:math:`\partial V_j / \partial \beta_k`, avoiding repeated Python tree
traversals inside the optimisation loop.

**Dummy-coded attributes**

Comparison operators return indicator expressions (1.0 / 0.0), not Python
booleans.  This enables dummy coding for categorical attributes directly in
the utility specification::

    # 3-level attribute: level 1 is reference, levels 2 and 3 get dummies
    beta_A_2 = Parameter('beta_A_2', 0.3)
    beta_A_3 = Parameter('beta_A_3', 0.6)

    V1 = beta_A_2 * (alt1_A == 2) + beta_A_3 * (alt1_A == 3)

Condition syntax
-----------------

Conditions are plain strings passed to
:meth:`~choicedesign.design.EffDesign.gen_initdesign` as a list.  They apply
during both initial design generation and the optimisation swaps — only
designs that satisfy all conditions are accepted.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Syntax
     - Meaning
   * - ``'alt1_cost > alt2_cost'``
     - Binary relation between two attributes or an attribute and a value.
   * - ``'if alt1_time > 20 then alt2_time < 30'``
     - Material implication: whenever the antecedent holds, the consequent
       must also hold.
   * - ``'alt1_cost > alt2_cost & alt1_time < alt2_time'``
     - Compound: all sub-conditions joined by ``&`` must hold simultaneously.
   * - ``'(alt1_A + alt1_B + alt1_C) > 0'``
     - Arithmetic expression on the left-hand side.  Any mix of attribute
       names and numeric constants combined with ``+``, ``-``, ``*``, ``/``
       and parentheses is valid on either side of the comparison.
   * - ``'if (alt1_A + alt1_B) > 0 then alt1_price >= 0'``
     - Arithmetic expression inside an ``if/then`` antecedent or consequent.

Arithmetic expressions can appear on either side of any comparison operator
and can be freely combined with ``if/then`` and ``&``.

Attribute names in condition strings must exactly match the ``name`` argument
of the corresponding :class:`~choicedesign.expressions.Attribute`.  A typo
raises a ``ValueError`` immediately when
:meth:`~choicedesign.design.EffDesign.gen_initdesign` is called.

Stopping criteria
------------------

At least one stopping criterion must be supplied to
:meth:`~choicedesign.design.EffDesign.optimise`.  They are checked after
every iteration and the first one to trigger stops the algorithm.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Argument
     - Meaning
   * - ``time_lim``
     - Stop after *N* minutes of wall-clock time.
   * - ``iter_lim``
     - Stop after *N* total iterations.
   * - ``noimprov_lim``
     - Stop after *N* consecutive iterations without improvement.

Optimisation algorithms
------------------------

Three algorithms are available via the ``algorithm`` argument of
:meth:`~choicedesign.design.EffDesign.optimise`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Description
   * - ``'swap'`` *(default)*
     - **Random Swapping** — picks a random attribute column and swaps the
       values of two randomly chosen rows.  Fast per iteration; good general
       default.
   * - ``'rsc'``
     - **RSC (Relabelling, Swapping, Cycling)** — applies one of three random
       column moves per iteration.  More diverse search than pure swapping.
   * - ``'federov'``
     - **Modified Federov** — replaces one row at a time with the best
       candidate from the full factorial.  More systematic but slower per
       iteration; works best for small attribute spaces.

Utility balance
----------------

After optimisation, the **utility balance ratio** measures how evenly the
prior parameters distribute expected market share across alternatives.
A ratio of 100 % means all alternatives have equal expected choice
probabilities; a ratio near 0 % indicates near-complete dominance by one
alternative.

Designs with very low utility balance may indicate that the prior values are
poorly calibrated or that the design is too constrained.
