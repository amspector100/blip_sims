#!/bin/sh
NREPS=1
NPROCESSES=1 

# we ran this simulation with k in {1,3,5}
ARGS="
	--reps $NREPS
	--num_processes $NPROCESSES
	--sparsity 0.05
	--coeff_size 1.5
	--kappa [0.2,0.4,0.6,0.8,1.0]
	--p 500
	--k 3
"

python3.9 weight_sensitivity.py $ARGS