#!/bin/sh
NREPS=1
NPROCESSES=1 

ARGS="
	--reps $NREPS
	--num_processes $NPROCESSES
	--p 1000
	--k 5
	--sparsity [0.05]
	--kappa [0.2,0.5,0.9]
	--nsample [200]
	--chains [1,2,3,5,7,10]
	--well_specified False
	--bsize 5
	--coeff_size 0.5
"

## Robustness to MCMC convergence plot
python3.9 convergence.py $ARGS