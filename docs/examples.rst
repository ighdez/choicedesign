Examples
========

The ``examples/`` directory contains Jupyter notebooks that demonstrate
different design scenarios.  Run them with::

    jupyter notebook examples/

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it demonstrates
   * - ``rum_simple``
     - Basic two-alternative linear utility design with blocking.
   * - ``rum_dummy``
     - Categorical attributes encoded with dummy variables.
   * - ``rum_asc``
     - Models with alternative-specific constants.
   * - ``rum_optout``
     - Three-alternative design including an unlabelled opt-out alternative.
   * - ``rum_avail``
     - Alternatives with different attribute subsets (availability conditions).
   * - ``rum_conds``
     - Attribute-level constraints using the condition syntax.
   * - ``rum_bayes``
     - Bayesian (Db-efficient) design with uncertain priors.

.. note::

   ``rum_bayes`` is currently experimental.  See :doc:`concepts` for details
   on Bayesian designs and the ``bayes_draws`` argument.
