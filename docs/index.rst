.. choicedesign documentation master file, created by
   sphinx-quickstart on Sun Nov  6 12:58:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ChoiceDesign's documentation!
========================================

ChoiceDesign is a Python package for generating **D-efficient designs** for
**Discrete Choice Experiments (DCE)**.  It provides a clean API for defining
utility functions symbolically, optimising designs under constraints, and
evaluating design quality with multiple optimality criteria.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   concepts
   examples

.. toctree::
   :maxdepth: 2
   :caption: API reference

.. autosummary::
   :toctree: generated

   choicedesign.design
   choicedesign.expressions
   choicedesign.criteria
   choicedesign.algorithms
   choicedesign.utils