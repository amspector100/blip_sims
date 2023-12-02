# BLiP Simulations

This repository contains all the code to replicate the experiments from [Spector and Janson (2022)](https://arxiv.org/pdf/2203.17208.pdf). Running these scripts requires a python environment with ``pyblip`` installed: please see https://github.com/amspector100/pyblip for installation.

Note that a variety of other code and software was published with the paper, which is listed below.

- pyblip, a Python package implementing BLiP: https://github.com/amspector100/pyblip.
- blipr, an R package implementing BLiP: https://github.com/amspector100/blipr.
- The code for the fine-mapping analysis: https://github.com/amspector100/ukbb_blip.
- The code for the astronomical application: https://github.com/amspector100/DeblendingStarfields.

## Overview

The directory ``blip_sims/`` contains extraneous functions used in the simulations. You should think of this as a supplemental package (for example, it contains code to run the distilled conditional randomization test).

The directory ``sims/`` contains the code which actually runs the simulations. It also contains ``.sh`` files which will replicate the figures.

The file ``final_sim_plots.ipynb`` is a jupyter notebook which contains (1) code to replicate the concrete examples in the paper and (2) the code used to generate the final plots for the paper. 

In particular, the main simulations in the paper can be replicated using the following scripts (their respective .sh files show how to use them). Note that the figure names line up with arXiv version v5 (to be released in the next 1-2 weeks).

- Figures 1, C.2, C.3: ``sims/lp_int_sol.sh``
- Figures 2, 3, 4, F.7, F.10, F.11, F.12, F.14: ``sims/glms.sh``
- Figures F.16, F.17, F.18: ``sims/changepoint.sh``
- Figure E.5: ``sims/susie_pips.sh``
- Figure F.13: ``sims/weight_sensitivity.sh``
- Figure E.4: ``sims/convergence.sh``
- Figures A.1, F.6, F.8, F.9: directly in ``final_sim_plots.ipynb``