# BLiP Simulations

This repository contains all the code to replicate the experiments from [Spector and Janson (2022)](https://arxiv.org/pdf/2203.17208.pdf). Running these scripts requires a python environment with ``pyblip`` installed: please see https://github.com/amspector100/pyblip for installation.

Note that a variety of other code and software was published with the paper, which is listed below.

- pyblip, a Python package implementing BLiP: https://github.com/amspector100/pyblip.
- blipr, an R package implementing BLiP: https://github.com/amspector100/blipr.
- The code for the fine-mapping analysis: https://github.com/amspector100/ukbb_blip.
- The code for the astronomical application: https://github.com/amspector100/DeblendingStarfields.

## Overview

The directory ``blip_sims/`` contains extraneous functions used in the simulations. You should think of this as a supplemental package (for example, it contains code to run the distilled conditional randomization test).

THe directory ``sims/`` contains the code which actually runs the simulations. It also contains ``.sh`` files which will replicate the figures.

In particular, the main figures in the paper can be replicated using the following scripts and their respective .sh files. Note that the figure names line up with the arXiv version.

Figures 1, 10, 11: ``sims/lp_int_sol.py``
Figures 2, 3, 4, 14: ``sims/glms.py``
Figures 16, 17, 18: ``sims/changepoint.py``