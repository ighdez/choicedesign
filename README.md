# ChoiceDesign

**ChoiceDesign** is a Python package tool to construct D-efficient designs for Discrete Choice Experiments. ChoiceDesign combines enough flexibility to construct from simple 2-alternative designs with few attributes, to more complex settings that may involve conditions between attributes. ChoiceDesign is a revamped version of [EDT](https://github.com/ighdez/EDT), a project I created some years ago for the same purpose. ChoiceDesign includes improvements over EDT such as class-based syntax, coding improvements, better documentation and making this package available to install via `pip`.

## Instalation

ChoiceDesign is available to install via the regular syntax of `pip`:

* ``python3 -m pip install choicedesign``

## Features

The main features of ChoiceDesign are:

* Allows to customize each attribute in terms of:
  * Attribute Levels
  * Continuous or Dummy coding (Effects coding is work-in-progress)
  * Assignement of prior parameters
  * Attribute names

* Designs with constraints: ChoiceDesign allows to define conditions over different attribute levels.
* Designs with blocks.
* Designs with alternative-specific constants (ASC).
* Multiple stopping criteria (Fixed number of iterations, iterations without improvement or fixed time).

Any contributions to ChoiceDesign are welcome via this Git, or to the email joseignaciohernandezh at gmail dot com. 

## References
* Kuhfeld, W. F. (2005). Experimental design, efficiency, coding, and choice designs. *Marketing research methods in SAS: Experimental design, choice, conjoint, and graphical techniques*, 47-97.
* Quan, W., Rose, J. M., Collins, A. T., & Bliemer, M. C. (2011). A comparison of algorithms for generating efficient choice experiments.