# BLiP Simulations (anonymized)

This repository contains all the code to replicate the experiments from the paper "Controlled Discovery and Localization of Signals via
Bayesian Linear Programming." Running these scripts requires a python environment with ``pyblip`` installed, which can be installed using
 ``python3.9 -m pip install -U pyblip.``

 Note that any figures from the real applications (Section 5 and Appendices G, H) can be reproduced using the other attached repositories.

## Overview

The directory ``blip_sims/`` contains extraneous functions used in the simulations. You should think of this as a supplemental package (for example, it contains code to run the distilled conditional randomization test).

The directory ``sims/`` contains the code which actually runs the simulations. It also contains ``.sh`` files which will replicate the figures.

The file ``final_sim_plots.ipynb`` is a jupyter notebook which contains (1) code to replicate the concrete examples in the paper and (2) the code used to generate the final plots for the paper. 

In particular, the main simulations in the paper can be replicated using the following .sh files:

- Figures 1, C.2, C.3: ``sims/lp_int_sol.sh``
- Figures 2, 3, 4, E.4, F.7, F.10, F.11, F.12, F.14: ``sims/glms.sh``
- Figures F.16, F.17, F.18: ``sims/changepoint.sh``
- Figure E.5: ``sims/susie_pips.sh``
- Figure F.13: ``sims/weight_sensitivity.sh``
- Figures A.1, F.6, F.8, F.9: directly in ``final_sim_plots.ipynb``

Many of these simulations are very computationally expensive, so you may need to use a computing cluster to run them.