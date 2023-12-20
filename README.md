# ChoiceDesign

**ChoiceDesign** is a Python package tool to construct D-efficient designs for Discrete Choice Experiments. ChoiceDesign combines enough flexibility to construct from simple 2-alternative designs with few attributes, to more complex settings that may involve conditions between attributes. ChoiceDesign is a revamped version of [EDT](https://github.com/ighdez/EDT), a project I created some years ago for the same purpose. ChoiceDesign includes improvements over EDT such as class-based syntax, coding improvements, better documentation and making this package available to install via `pip`.

**NEW:** ChoiceDesign now integrates [Biogeme](https://biogeme.epfl.ch/), which allows you to customise the utility functions.

## Installation

ChoiceDesign is available to install via the regular syntax of `pip`:

* ``python3 -m pip install choicedesign``

## Features

The main features of ChoiceDesign are:

* D-efficient designs based on a random swapping algorithm
* **Customisable utility functions** (powered by [Biogeme](https://biogeme.epfl.ch/))
* Bayesian priors *(experimental)*
* Designs with conditions over different attribute levels
* Designs with blocks.
* Multiple stopping criteria (Fixed number of iterations, iterations without improvement or fixed time).

## Examples

I provide some Jupyter notebooks that illustrate the use of ChoiceDesign in the `examples/` folder of this repo.

## How to contribute?
Any contributions to ChoiceDesign are welcome via this Git, or to my email joseignaciohernandezh at gmail dot com.

## Disclaimer

This software is provided for free and as it is, say with **no warranty**, and neither me nor my current institution is liable of any consequence of the use of it. In any case, integrity checks have been performed by comparing results with alternative software.

## References
* Bierlaire, M. (2003). BIOGEME: A free package for the estimation of discrete choice models. In *Swiss transport research conference*.
* Kuhfeld, W. F. (2005). Experimental design, efficiency, coding, and choice designs. *Marketing research methods in SAS: Experimental design, choice, conjoint, and graphical techniques*, 47-97.
* Quan, W., Rose, J. M., Collins, A. T., & Bliemer, M. C. (2011). A comparison of algorithms for generating efficient choice experiments.