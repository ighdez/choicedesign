# ChoiceDesign

![Tests](https://github.com/ighdez/choicedesign/actions/workflows/tests.yml/badge.svg)

**ChoiceDesign** is a Python package tool to construct efficient designs for Discrete Choice Experiments. ChoiceDesign combines enough flexibility to construct from simple 2-alternative designs with few attributes, to more complex settings that may involve conditions between attributes. ChoiceDesign is a revamped version of [EDT](https://github.com/ighdez/EDT), a project I created some years ago for the same purpose. ChoiceDesign includes improvements over EDT such as class-based syntax, coding improvements, better documentation and making this package available to install via `pip`.

## Installation

ChoiceDesign is available to install via the regular syntax of `pip`:

* ``python3 -m pip install choicedesign``

## Features

The main features of ChoiceDesign are:

* **D-efficient, A-efficient, and C-efficient designs** — minimise D-error (determinant), A-error (average variance), or C-error (WTP variance sum)
* **Db-efficient (Bayesian) designs** via `Parameter(prior_std=...)` and `bayes_draws` argument
* **Three optimisation algorithms** — random swapping, RSC (Relabelling, Swapping, Cycling), and Modified Federov
* **Customisable utility functions** (no external dependencies — expressions rewritten from scratch)
* Designs with conditions over different attribute levels
* Designs with blocks
* Full-factorial designs
* Multiple stopping criteria (fixed number of iterations, iterations without improvement, or fixed time)

## Examples

I provide some Jupyter notebooks that illustrate the use of ChoiceDesign in the `examples/` folder of this repo.

## How to contribute?
Any contributions to ChoiceDesign are welcome via this Git, or to my email joseignaciohernandezh at gmail dot com.

## Acknowledgements

Parts of this project were developed with the assistance of [Claude](https://claude.ai) by Anthropic.

## Disclaimer

This software is provided for free and as it is, say with **no warranty**, and neither me nor my current institution is liable of any consequence of the use of it. In any case, integrity checks have been performed by comparing results with alternative software.

## References
* Bierlaire, M. (2003). BIOGEME: A free package for the estimation of discrete choice models. In *Swiss transport research conference*.
* Kuhfeld, W. F. (2005). Experimental design, efficiency, coding, and choice designs. *Marketing research methods in SAS: Experimental design, choice, conjoint, and graphical techniques*, 47-97.
* Quan, W., Rose, J. M., Collins, A. T., & Bliemer, M. C. (2011). A comparison of algorithms for generating efficient choice experiments.