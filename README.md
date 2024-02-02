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

In particular, the main simulations in the paper can be replicated using the following scripts (their respective .sh files show how to use them). 

### Replication for JASA figure numbering

- Figures 1, C.2, C.3: ``sims/lp_int_sol.sh``
- Figures 2, 3, 4, F.6, F.9, F.10, F.11, F.12, F.13, F.14, F.16: ``sims/glms.sh``
- Figures F.17, F.18, F.19, F.20: ``sims/changepoint.sh``
- Figure E.4: ``sims/susie_pips.sh``
- Figure F.15: ``sims/weight_sensitivity.sh``
- Figure 5: ``sims/convergence.sh``
- Figures A.1, F.5, F.7, F.8: directly in ``final_sim_plots.ipynb``

Any figures not listed here can be reproduced in https://github.com/amspector100/ukbb_blip (for the fine-mapping analysis) or https://github.com/amspector100/DeblendingStarfields (for the astronomical application).

### Replication for arXiv figure numbering

- Figures 1, 10, 11: ``sims/lp_int_sol.sh``
- Figures 2, 3, 4, 15: ``sims/glms.sh``
- Figures 17, 18, 19, 20: ``sims/changepoint.sh``
- Figure 13: ``sims/susie_pips.sh``
- Figure 16: ``sims/weight_sensitivity.sh``
- Figure 12: ``sims/convergence.sh``
- Figures 9, 14: directly in ``final_sim_plots.ipynb``

Any figures not listed here can be reproduced in https://github.com/amspector100/ukbb_blip (for the fine-mapping analysis) or https://github.com/amspector100/DeblendingStarfields (for the astronomical application).